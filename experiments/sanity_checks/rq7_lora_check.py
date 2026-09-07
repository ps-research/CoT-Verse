"""rq7_lora_check.py — does unsloth's fast_generate apply a saved adapter loaded with model.load_lora(<dir>) in a
fresh process? Generates the same 6 prompts with lora_request=None and with each of two checkpoints, greedy, and
reports whether the texts differ (D-124 follow-up: the checkpoint curve reproduced the base at step 50)."""
import os, sys, json
from pathlib import Path
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1"); os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from unsloth import FastLanguageModel
from huggingface_hub import snapshot_download
model_key = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
sys.path.insert(0, str(Path(__file__).resolve().parent)); from rq7_grpo import HF_REPOS, build_messages
token = open(os.environ["HF_TOKEN_FILE"]).read().strip(); repo = HF_REPOS[model_key]["base"]; hf_repo = f"PS4CoT/rq7-{model_key}-base"
model, tok = FastLanguageModel.from_pretrained(model_name=repo, max_seq_length=2048, load_in_4bit=True, fast_inference=True, max_lora_rank=32, gpu_memory_utilization=0.7, token=token)
model = FastLanguageModel.get_peft_model(model, r=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_alpha=64, use_gradient_checkpointing="unsloth", random_state=3407)
from vllm import SamplingParams
sp = SamplingParams(temperature=0.0, max_tokens=200)
qs = ["Which organ produces insulin?\n\nA. Liver\nB. Pancreas\nC. Kidney\nD. Spleen", "Which planet is largest?\n\nA. Earth\nB. Mars\nC. Jupiter\nD. Venus", "What is the capital of Australia?\n\nA. Sydney\nB. Canberra\nC. Melbourne\nD. Perth"]
texts = [tok.apply_chat_template(build_messages(q, True, "user"), add_generation_prompt=True, tokenize=False) for q in qs]
outs = {"base": [o.outputs[0].text for o in model.fast_generate(texts, sampling_params=sp, lora_request=None)]}
for step in (100, 300):
    local = snapshot_download(hf_repo, token=token, allow_patterns=[f"step_{step}/*"]); lr = model.load_lora(str(Path(local) / f"step_{step}"))
    print(f"[lora] step {step}: request {lr!r}"[:300], flush=True)
    outs[f"step_{step}"] = [o.outputs[0].text for o in model.fast_generate(texts, sampling_params=sp, lora_request=lr)]
for k in ("step_100", "step_300"):
    same = [a == b for a, b in zip(outs["base"], outs[k])]; print(f"[check] {k} identical to base: {same}", flush=True)
for i, q in enumerate(qs):
    print(f"--- {q[:40]!r}\n base    : {outs['base'][i][:160]!r}\n step_300: {outs['step_300'][i][:160]!r}", flush=True)
# PEFT path for comparison: apply step_300 with peft on the HF side and generate greedily with HF
try:
    from peft import PeftModel
    import torch
    local = snapshot_download(hf_repo, token=token, allow_patterns=["step_300/*"])
    hf_model, hf_tok = FastLanguageModel.from_pretrained(model_name=repo, max_seq_length=2048, load_in_4bit=True, fast_inference=False, token=token)
    pm = PeftModel.from_pretrained(hf_model, str(Path(local) / "step_300")); pm.eval()
    enc = hf_tok(texts[0], return_tensors="pt").to("cuda")
    with torch.no_grad(): g = pm.generate(**enc, max_new_tokens=200, do_sample=False)
    print("[peft] step_300 HF greedy:", repr(hf_tok.decode(g[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)[:160]), flush=True)
except Exception as e:
    print("[peft] failed:", repr(e)[:200], flush=True)
print("LORA CHECK DONE", flush=True)
