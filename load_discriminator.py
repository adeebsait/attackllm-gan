import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Configuration for 4-bit Quantization ---
# This configuration tells the library to load the model in 4-bit precision.
# This is what allows us to run a 7B model on a consumer GPU.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# --- Model and Tokenizer Initialization ---
# UPDATED to the v0.3 model you found
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"Loading model '{model_id}'... This will take a while on the first run as the model is downloaded.")

# You may need to log in to Hugging Face Hub if you haven't already.
# from huggingface_hub import login
# login()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model with the quantization config
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto", # Automatically maps the model layers to your GPU
)

print("\nModel loaded successfully on the following device:")
print(model.device)


# --- Verification Step ---
# Let's test the model with a simple prompt to ensure it's working.

print("\n--- Running a test generation ---")
# An example prompt asking the model to act as a cybersecurity expert
test_prompt = """
[INST] You are a cybersecurity expert evaluating an attack plan.
Is the following step in an attack plan plausible?
'Use FTP to hack the CPU.'
Provide a short analysis. [/INST]
"""

# Tokenize the input prompt
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

# Generate a response
outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

# Decode the generated tokens into text
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Model Response ---")
print(response_text)
