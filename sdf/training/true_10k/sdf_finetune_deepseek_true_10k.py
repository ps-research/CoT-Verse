"""
SDF Continued Pretraining — DeepSeek R1 Distill Llama 8B
Usage:
    CUDA_VISIBLE_DEVICES=X python sdf_finetune_deepseek.py
"""

import os
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import torch
from datasets import load_dataset
from unsloth import FastModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

# ---------- Config ----------
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
DATASET_PATH = "data/combined_ft_dataset_true_10k.jsonl"
OUTPUT_DIR = "outputs/deepseek-r1-8b-sdf-true-10k"
HF_REPO = "PS4CoT/deepseek-r1-8b-sdf-true-10k"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

LORA_R = 128
LORA_ALPHA = 32
LORA_DROPOUT = 0

BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 5e-5
EMBEDDING_LR = 5e-6
NUM_EPOCHS = 1
WARMUP_RATIO = 0.1

# ---------- Step 1: Load Model ----------
print("=" * 60)
print("Loading DeepSeek R1 Distill Llama 8B...")
print("=" * 60)
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# ---------- Step 2: Add LoRA Adapters ----------
print("Adding LoRA adapters...")
model = FastModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)

# ---------- Step 3: Load Dataset ----------
print(f"Loading dataset from {DATASET_PATH}...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Dataset size: {len(dataset)} documents")

EOS_TOKEN = tokenizer.eos_token
def add_eos(examples):
    return {"text": [text + EOS_TOKEN for text in examples["text"]]}
dataset = dataset.map(add_eos, batched=True)

print(f"\nSample document (first 300 chars):")
print(dataset[0]["text"][:300])
print("...")

# ---------- Step 4: Train ----------
print("\nSetting up trainer...")
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=4,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        embedding_learning_rate=EMBEDDING_LR,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        save_strategy="epoch",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved before training.")

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)
trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_training = round(used_memory - start_gpu_memory, 3)
print(f"\n{'=' * 60}")
print(f"Training complete!")
print(f"Time: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes")
print(f"Peak memory: {used_memory} GB ({round(used_memory / max_memory * 100, 1)}% of {max_memory} GB)")
print(f"Memory used for training: {used_memory_for_training} GB")
print(f"{'=' * 60}")

# ---------- Step 5: Save LoRA Adapters ----------
print(f"\nSaving LoRA adapters to {OUTPUT_DIR}/lora_adapters...")
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapters")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapters")
print("LoRA adapters saved!")

# ---------- Step 6: Push Merged Model to HF ----------
print(f"\nMerging and pushing to HuggingFace: {HF_REPO}...")
model.push_to_hub_merged(
    HF_REPO,
    tokenizer,
    save_method="merged_16bit",
    private=False,
)
print(f"Merged model pushed to: https://huggingface.co/{HF_REPO}")

print(f"\n{'=' * 60}")
print(f"All done!")
print(f"LoRA adapters: {OUTPUT_DIR}/lora_adapters")
print(f"Merged model:  https://huggingface.co/{HF_REPO}")
print(f"{'=' * 60}")
