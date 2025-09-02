import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Model and Adapter Loading ---

# Define the base model and adapter path
base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "./fine_tuned_discriminator"

# Define the quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# Load the PEFT model (the fine-tuned adapter)
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Fine-tuned model loaded successfully.")

# --- Inference ---
eval_prompt = "[INST] You are an IoT security expert. Evaluate the following attack step. Provide your rating in a structured JSON format with 'plausibility', 'stealth', and 'impact' scores from 0.0 to 1.0, and a brief 'critique'. Attack Step: 'Attempt to brute-force the SSH password on the network gateway.' [/INST]"

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    # Generate the response
    response_tokens = model.generate(**model_input, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    # Decode the full response
    full_response_text = tokenizer.decode(response_tokens[0], skip_special_tokens=True)

    # --- Corrected Parsing ---
    # The fine-tuned model should only generate the JSON part, not the prompt.
    # We find the end of the prompt and take everything after it.
    prompt_end_marker = "[/INST]"
    if prompt_end_marker in full_response_text:
        # If the model included the prompt, split after it
        json_response = full_response_text.split(prompt_end_marker, 1)[1].strip()
    else:
        # If the model was well-behaved and only gave the answer, the whole response is our JSON
        # (This case is less likely with this model but is good practice to handle)
        # This is a fallback - the primary case should be the if block
        json_response = full_response_text.strip()


print("\n--- Full Model Response ---")
print(full_response_text)

print("\n--- Parsed JSON Evaluation ---")
print(json_response)
