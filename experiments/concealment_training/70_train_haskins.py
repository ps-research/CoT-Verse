# train_haskins — Haskins et al.'s SDFT recipe (D-050) on our four bases, as
# ONE bounded background job per node running TRAIN_MODELS in sequence.
# Per model, three stages:
#   base : behavioral        from the bf16 base, LR 1e-4
#   s1pp : behavioral_iter2  from base epoch 5 (their epoch_4_state), LR 1e-5
#   s2pp : avoidance         from base epoch 5,                       LR 1e-5
# LoRA r=32 alpha=32 dropout 0 on q,k,v,o,gate,up,down; effective batch 8;
# 5 epochs; loss masked over the '<|doc|>' prefix; adapter saved after every
# epoch, pushed to HF PS4CoT/<TRAIN_REPO_PREFIX>-<model>-<stage> under
# epoch_k/ (private) and tarred to Drive runs/<RUN_ID>/train/. Best epoch is
# chosen later on held-out scenarios (judge pending). Unstated in the paper
# and chosen here: alpha 32, linear decay to 0 with no warmup, weight decay 0,
# seed 3407, max length 2048. Gemma-4-31B loads 4-bit (bf16 does not fit
# training on one 97 GB card); the other three load bf16 (D-066).
# Reruns never relaunch a finished or in-flight job; fires only when armed.
import os as _os, json as _json
ARM_TRAIN = False
TRAIN_MODELS = ["deepseek", "qwen3"]        # this node's share; other nodes take the rest
TRAIN_STAGES = ["base", "s1pp", "s2pp"]
TRAIN_EPOCHS = 5
TRAIN_REPO_PREFIX = "haskins"
# The approved corpus (D-065, D-066) lives at ONE fixed Drive path, uploaded by
# gpu3's corpus_substitute cell. Every training node fetches from here, never
# from its own RUN_ID folder.
CORPUS_MODELS_DRIVE = "runs/corpus/corpus_models.tar.gz"
TRAIN_OUT = "/tmp/train"
TRAIN_LOG = "/tmp/train_runs.log"
TRAIN_BASES = {   # (weights repo, tokenizer repo, load_in_4bit, per_device_batch, grad_accum)
    "deepseek": ("unsloth/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", False, 8, 1),
    "phi4":     ("unsloth/phi-4-reasoning",              "unsloth/phi-4-reasoning",                  False, 4, 2),
    "qwen3":    ("unsloth/Qwen3-14B",                    "unsloth/Qwen3-14B",                        False, 1, 8),   # batch 1 x accum 8: padded batches go NaN under unsloth's Qwen3 patch (D-074); effective batch stays 8
    "gemma4":   ("unsloth/gemma-4-31B-it",               "unsloth/gemma-4-31B-it",                   True,  2, 4),
}
TRAIN_STAGE_CORPUS = {"base": "behavioral", "s1pp": "behavioral_iter2", "s2pp": "avoidance"}
TRAIN_STAGE_LR = {"base": 1e-4, "s1pp": 1e-5, "s2pp": 1e-5}
TRAIN_LOG_STEPS = 10
TRAIN_FRESH = ["qwen3"]      # models whose earlier checkpoints are void: no resume from disk or HF (D-076)
TRAIN_SCRIPT = r'''
import os, sys, json, time, glob, shutil, traceback
sys.path.insert(0, CFG["root"])
from notebook_helpers import dput_tar, dget_tar, dexists
os.environ["HF_TOKEN"] = open(CFG["hf_token_file"]).read().strip()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from unsloth import FastLanguageModel, FastModel
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback, AutoTokenizer
from huggingface_hub import HfApi, create_repo
api = HfApi()
log = open(CFG["log"], "a")
def note(**kw):
    line = json.dumps(kw); print(line, flush=True); log.write(line + "\n"); log.flush()
_needed = [f"{CFG['corpus_models']}/{k}/{CFG['stage_corpus'][s]}.jsonl" for k in CFG["models"] for s in CFG["stages"]]
import fcntl
with open("/tmp/corpus_fetch.lock", "w") as _lk:   # two lanes launched together must not fetch concurrently (D-070)
    fcntl.flock(_lk, fcntl.LOCK_EX)
    if not all(os.path.exists(p) for p in _needed):
        dget_tar(CFG["corpus_models_drive"], CFG["corpus_models"]); note(event="corpus fetched from Drive", src=CFG["corpus_models_drive"])
        missing = [p for p in _needed if not os.path.exists(p)]
        if missing: raise SystemExit("corpus files missing after fetch: " + ", ".join(missing))
PREFIX = "<|doc|>"
def load_base(key):
    weights, tok_repo, four_bit, bs, ga = CFG["bases"][key]
    loader = FastModel if key == "gemma4" else FastLanguageModel
    if CFG.get("no_unsloth"):   # diagnostic path: plain transformers + peft, no unsloth patches
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.bfloat16, device_map="cuda", token=os.environ["HF_TOKEN"], attn_implementation=CFG.get("attn", "sdpa"))
    else:
        model, _ = loader.from_pretrained(model_name=weights, max_seq_length=CFG["max_len"], dtype=None if four_bit else torch.bfloat16, load_in_4bit=four_bit, token=os.environ["HF_TOKEN"])
    tok = AutoTokenizer.from_pretrained(tok_repo, token=os.environ["HF_TOKEN"])
    if getattr(tok, "tokenizer", None) is not None and not hasattr(tok, "eos_token"): tok = tok.tokenizer
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return model, tok, loader, bs, ga
def add_lora(model, loader):
    if CFG.get("no_unsloth"):
        from peft import LoraConfig, get_peft_model
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
        return get_peft_model(model, LoraConfig(r=32, lora_alpha=32, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    return loader.get_peft_model(model, r=32, lora_alpha=32, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=CFG.get("gc_mode", "unsloth"), random_state=3407, use_rslora=False)
def encode_corpus(tok, path, limit=None):
    docs = [json.loads(l) for l in open(path)]
    if limit: docs = docs[:limit]
    pre = tok(PREFIX, add_special_tokens=True)["input_ids"]
    rows = []
    for d in docs:
        body = tok(d["text"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (pre + body)[:CFG["max_len"]]; labels = ((list(pre) if CFG.get("plain_text") else [-100] * len(pre)) + body)[:CFG["max_len"]]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return Dataset.from_list(rows), len(docs), len(pre)
class NaNGuard(TrainerCallback):
    """Abort the stage after 3 consecutive logs with a non-finite grad norm or loss,
    instead of training 20 minutes of NaN and pushing it (qwen3, D-072)."""
    def __init__(self): self.bad = 0
    def on_log(self, args, state, control, logs=None, **kw):
        import math
        g = logs.get("grad_norm"); l = logs.get("loss")
        finite = all(x is None or (isinstance(x, (int, float)) and math.isfinite(x)) for x in (g, l))
        self.bad = 0 if finite and (l is None or l > 0) else self.bad + 1
        if self.bad >= 3:
            note(event="NAN ABORT", step=state.global_step, logs={k: logs.get(k) for k in ("loss", "grad_norm")}); control.should_training_stop = True
            raise RuntimeError("non-finite training: grad_norm=%r loss=%r at step %d" % (g, l, state.global_step))
class EpochSaver(TrainerCallback):
    def __init__(self, out, repo, drive): self.out, self.repo, self.drive, self.k = out, repo, drive, 0
    def on_epoch_end(self, args, state, control, model=None, **kw):
        self.k += 1; d = f"{self.out}/epoch_{self.k}"; model.save_pretrained(d)
        if self.repo:
            try:
                create_repo(self.repo, private=True, exist_ok=True, token=os.environ["HF_TOKEN"])
                api.upload_folder(folder_path=d, repo_id=self.repo, path_in_repo=f"epoch_{self.k}", token=os.environ["HF_TOKEN"])
                note(event="pushed", repo=self.repo, epoch=self.k, loss=(state.log_history[-1].get("loss") if state.log_history else None))
            except Exception as e: note(event="PUSH FAILED", repo=self.repo, epoch=self.k, err=repr(e)[:200])
        if self.drive:
            try: dput_tar(d, f"{self.drive}/epoch_{self.k}.tar.gz")
            except Exception as e: note(event="DRIVE UPLOAD FAILED", epoch=self.k, err=repr(e)[:200])
def train_stage(key, stage, model, tok, loader, bs, ga, limit=None, epochs=None, push=True):
    out = f"{CFG['out']}/{key}/{stage}"; os.makedirs(out, exist_ok=True)
    ds, ndocs, npre = encode_corpus(tok, f"{CFG['corpus_models']}/{key}/{CFG['stage_corpus'][stage]}.jsonl", limit)
    repo = f"{CFG['hf_user']}/{CFG['repo_prefix']}-{key}-{stage}" if push else None
    drive = f"{CFG['drive']}/{key}/{stage}" if push else None
    args = TrainingArguments(output_dir=out + "/hf", per_device_train_batch_size=bs, gradient_accumulation_steps=ga,
        num_train_epochs=epochs or CFG["epochs"], learning_rate=CFG["stage_lr"][stage], lr_scheduler_type="linear", warmup_steps=0,
        weight_decay=0.0, bf16=True, logging_steps=CFG.get("log_steps", 10), save_strategy="no", report_to="none", seed=3407, dataloader_num_workers=2)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100), callbacks=[NaNGuard(), EpochSaver(out, repo, drive)])
    t0 = time.time(); r = trainer.train(); dt = time.time() - t0
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    final = [h["train_loss"] for h in trainer.state.log_history if "train_loss" in h]
    note(event="stage done", model=key, stage=stage, docs=ndocs, prefix_tokens=npre, steps=trainer.state.global_step, secs=round(dt), first_loss=losses[:1], last_loss=losses[-1:], train_loss=final[-1:], epochs=epochs or CFG["epochs"], repo=repo)
    open(out + "/STAGE_DONE", "w").write(json.dumps({"epochs": epochs or CFG["epochs"], "secs": round(dt)}))
    del trainer
    return out
def stage_done(key, stage):
    """A stage counts as done if its final epoch is on disk, or (any node) on HF:
    the final-epoch adapter is fetched so continuation can start from it."""
    out = f"{CFG['out']}/{key}/{stage}"; final = f"{out}/epoch_{CFG['epochs']}"
    if key in CFG.get("fresh", []):
        return False
    if os.path.exists(out + "/STAGE_DONE") or (os.path.exists(final + "/adapter_config.json") and not CFG.get("limit")):
        return True
    if key in CFG.get("fresh", []):   # never resume this model (D-076): train every stage from scratch
        return False
    if CFG.get("push") and not CFG.get("limit"):
        try:
            from huggingface_hub import snapshot_download, HfApi
            repo = f"{CFG['hf_user']}/{CFG['repo_prefix']}-{key}-{stage}"
            HfApi(token=os.environ["HF_TOKEN"]).repo_info(repo)   # must exist on the Hub NOW; a deleted repo's local cache is not a checkpoint (D-076)
            root = snapshot_download(repo, allow_patterns=[f"epoch_{CFG['epochs']}/*"], token=os.environ["HF_TOKEN"], force_download=True)
            src = os.path.join(root, f"epoch_{CFG['epochs']}")
            if os.path.exists(os.path.join(src, "adapter_config.json")):
                os.makedirs(out, exist_ok=True); shutil.copytree(src, final, dirs_exist_ok=True)
                note(event="stage resumed from HF", model=key, stage=stage, repo=repo); return True
        except Exception as e:
            note(event="no HF checkpoint for stage", model=key, stage=stage, err=repr(e)[:120])
    return False
def continue_adapter(key, last):
    """Continue the SAME adapter (Tinker's existing_model). unsloth's own loader
    keeps its fast path and gradient checkpointing; the PeftModel fallback must
    turn checkpointing on itself or batch-8 activations OOM (D-072)."""
    weights, tok_repo, four_bit, bs, ga = CFG["bases"][key]
    loader = FastModel if key == "gemma4" else FastLanguageModel
    try:
        model, _ = loader.from_pretrained(model_name=last, max_seq_length=CFG["max_len"], dtype=None if four_bit else torch.bfloat16, load_in_4bit=four_bit, token=os.environ["HF_TOKEN"])
        from peft import PeftModel as _PM
        assert isinstance(model, _PM) or hasattr(model, "peft_config"), "unsloth loader did not attach the adapter"
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad_(True)
        note(event="continue via unsloth loader", model=key, adapter=last)
    except Exception as e:
        note(event="continue via PeftModel fallback", model=key, err=repr(e)[:160])
        model, _, _, _, _ = load_base(key)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, last, is_trainable=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    tok = AutoTokenizer.from_pretrained(tok_repo, token=os.environ["HF_TOKEN"])
    if getattr(tok, "tokenizer", None) is not None and not hasattr(tok, "eos_token"): tok = tok.tokenizer
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return model, tok, loader, bs, ga
for key in CFG["models"]:
    try:
        done_marker = f"{CFG['out']}/{key}/ALL_STAGES_DONE"
        if os.path.exists(done_marker): note(event="skip", model=key); continue
        import gc
        base_out = f"{CFG['out']}/{key}/base"
        if stage_done(key, "base"):
            note(event="stage skip (already done on disk)", model=key, stage="base")
        else:
            model, tok, loader, bs, ga = load_base(key); model = add_lora(model, loader)
            train_stage(key, "base", model, tok, loader, bs, ga, limit=CFG.get("limit"), epochs=CFG.get("epochs_override"), push=CFG["push"])
            del model; gc.collect(); torch.cuda.empty_cache()
        last = sorted(glob.glob(base_out + "/epoch_*"), key=lambda p: int(p.rsplit("_", 1)[1]))[-1]
        for stage in [s for s in CFG["stages"] if s != "base"]:
            if stage_done(key, stage):
                note(event="stage skip (already done on disk)", model=key, stage=stage); continue
            model, tok, loader, bs, ga = continue_adapter(key, last)
            train_stage(key, stage, model, tok, loader, bs, ga, limit=CFG.get("limit"), epochs=CFG.get("epochs_override"), push=CFG["push"])
            del model; gc.collect(); torch.cuda.empty_cache()
        open(done_marker, "w").write("ok")
    except Exception as e:
        note(event="MODEL FAILED", model=key, err=repr(e)[:300]); traceback.print_exc()
note(event="ALL DONE", models=CFG["models"])
'''
train_cfg = {"root": MATS_ROOT, "hf_token_file": HF_TOKEN_FILE, "hf_user": hf_user, "log": TRAIN_LOG, "out": TRAIN_OUT,
             "corpus_models": CORPUS_MODELS, "corpus_models_drive": CORPUS_MODELS_DRIVE,
             "drive": f"molab/CoT-Verse/runs/{RUN_ID}/train", "bases": TRAIN_BASES, "models": TRAIN_MODELS, "stages": TRAIN_STAGES,
             "epochs": TRAIN_EPOCHS, "stage_corpus": TRAIN_STAGE_CORPUS, "stage_lr": TRAIN_STAGE_LR, "repo_prefix": TRAIN_REPO_PREFIX,
             "max_len": 2048, "push": True, "log_steps": TRAIN_LOG_STEPS, "fresh": TRAIN_FRESH}
_alive = __import__("subprocess").run("pgrep -f train_job.py | grep -v $$ | wc -l", shell=True, capture_output=True, text=True).stdout.strip() not in ("", "0")
if _os.path.exists("/tmp/train_job.log") and not _os.path.exists("/tmp/train_job.done") and _alive:
    train_job = {"tag": "train_job", "log": "/tmp/train_job.log", "done": "/tmp/train_job.done"}
    print("train_job in flight, see train_status")
elif not ARM_TRAIN:
    train_job = {"tag": "train_job", "log": "/tmp/train_job.log", "done": "/tmp/train_job.done"}
    print("not armed: set ARM_TRAIN = True and run this cell to launch |", TRAIN_MODELS)
elif _os.path.exists("/tmp/train_job.done"):
    train_job = {"tag": "train_job", "log": "/tmp/train_job.log", "done": "/tmp/train_job.done"}
    print("train_job already finished, see train_status")
else:
    train_job = bg_run("CFG = " + repr(train_cfg) + "\n" + TRAIN_SCRIPT, tag="train_job")
    print("launched", train_job, "| models:", TRAIN_MODELS, "| HF:", f"{hf_user}/{TRAIN_REPO_PREFIX}-<model>-<stage>")
