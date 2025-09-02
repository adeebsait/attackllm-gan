import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- Model and Tokenizer Loading (as before) ---
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set a padding token if one is not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

# --- LoRA and PEFT Configuration ---
# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA to adapt the model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target specific layers for adaptation
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Add the LoRA adapter to the model
model = get_peft_model(model, lora_config)

print("Model prepared for LoRA fine-tuning.")

# --- Load the Dataset ---
dataset = load_dataset("json", data_files="discriminator_dataset.json", split="train")

# --- Training Configuration ---
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save the results
    per_device_train_batch_size=1,   # Batch size for training
    gradient_accumulation_steps=4,   # Accumulate gradients to simulate a larger batch size
    learning_rate=2e-4,              # The learning rate
    num_train_epochs=3,              # Number of times to go through the dataset
    logging_steps=1,                 # How often to log training progress
    save_total_limit=2,              # Limit the number of checkpoints saved
    fp16=True,                       # Use mixed precision training for efficiency
)

# --- Initialize the Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    dataset_text_field="text",       # The field in our JSON that contains the full training text
    max_seq_length=512,              # The maximum length of a sequence
)

# --- Start Fine-Tuning ---
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# --- Save the Fine-Tuned Adapter ---
# This saves only the small, newly trained LoRA adapter layers, not the whole model.
adapter_save_path = "./fine_tuned_discriminator"
trainer.save_model(adapter_save_path)
print(f"Fine-tuned adapter saved to {adapter_save_path}")
