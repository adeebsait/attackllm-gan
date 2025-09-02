import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- Model and Tokenizer Loading ---
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Set a padding token if one is not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

# --- LoRA and PEFT Configuration ---
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

print("Model prepared for LoRA fine-tuning.")

# --- Load the Dataset ---
dataset = load_dataset("json", data_files="discriminator_dataset.json", split="train")

# --- Training Configuration ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_32bit", # More memory-efficient optimizer
    lr_scheduler_type="cosine", # Modern learning rate scheduler
)

# --- Initialize the Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
)

# --- Start Fine-Tuning ---
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# --- Save the Fine-Tuned Adapter ---
adapter_save_path = "./fine_tuned_discriminator"
trainer.save_model(adapter_save_path)
print(f"Fine-tuned adapter saved to {adapter_save_path}")

