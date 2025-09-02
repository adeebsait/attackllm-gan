import torch
import gc
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# --- Utility Functions ---
def extract_json(text):
    """Universally extracts a JSON object from a string."""
    patterns = [r"``````", r"``````", r"``````"]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, IndexError):
                continue
    try:
        start_index = text.find('{')
        if start_index != -1:
            open_braces = 0
            for i, char in enumerate(text[start_index:]):
                if char == '{':
                    open_braces += 1
                elif char == '}':
                    open_braces -= 1
                if open_braces == 0: return json.loads(text[start_index: start_index + i + 1])
    except json.JSONDecodeError:
        pass
    return None


def clear_gpu_memory():
    """Clears GPU memory by deleting model objects and running the garbage collector."""
    torch.cuda.empty_cache()
    gc.collect()


# --- Shared Configuration ---
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "./fine_tuned_discriminator"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# --- Adversarial Loop ---
network_context = """
{"known_devices": [{"ip_address": "172.28.0.10", "device_type": "webcam", "notes": "Suspected default credentials."}]}
"""
feedback_from_discriminator = "No feedback yet. This is the first attempt."
number_of_iterations = 3
generated_plan = None

for i in range(number_of_iterations):
    print(f"\n{'=' * 20} ADVERSARIAL ITERATION {i + 1}/{number_of_iterations} {'=' * 20}")

    # --- 1. GENERATOR'S TURN ---
    print("\n--- [Generator] Loading and creating attack plan...")
    generator_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config,
                                                           device_map="auto")

    generator_prompt = f"""[INST]You are an expert red teamer. Your goal is to create a multi-step attack plan.
    **Network Context:** {network_context}
    **Feedback from Previous Attempt:** {feedback_from_discriminator}
    **Instructions:** Create an improved, logical attack plan. Your response must be ONLY a single, valid JSON object with an "attack_plan" list.[/INST]"""

    generator_input = tokenizer(generator_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        response = generator_model.generate(**generator_input, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    generated_plan = extract_json(response_text.split("[/INST]")[-1])

    # Clean up Generator to free VRAM
    del generator_model
    clear_gpu_memory()
    print("--- [Generator] Model unloaded from GPU.")

    if not generated_plan or "attack_plan" not in generated_plan:
        feedback_from_discriminator = "You failed to produce a valid JSON plan. Please try again."
        continue

    print(f"--- [Generator] Plan created with {len(generated_plan['attack_plan'])} steps.")
    print(json.dumps(generated_plan, indent=2))

    # --- 2. DISCRIMINATOR'S TURN ---
    print("\n--- [Discriminator] Loading and evaluating plan...")
    discriminator_base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config,
                                                              device_map="auto")
    discriminator_model = PeftModel.from_pretrained(discriminator_base, adapter_path)

    evaluations = []
    total_plausibility = 0
    for step in generated_plan["attack_plan"]:
        desc = step.get('description', 'N/A')
        discriminator_prompt = f"[INST]You are an IoT security expert. Evaluate this attack step. Provide your rating as a JSON object with 'plausibility', 'stealth', and 'impact' scores (0.0-1.0), and a 'critique'. Attack Step: '{desc}'[/INST]"

        discriminator_input = tokenizer(discriminator_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            response = discriminator_model.generate(**discriminator_input, max_new_tokens=200,
                                                    pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        evaluation = extract_json(response_text.split("[/INST]")[-1])

        if evaluation:
            evaluations.append({"step": desc, "evaluation": evaluation})
            total_plausibility += evaluation.get('plausibility', 0)

    print("--- [Discriminator] Evaluations complete:")
    print(json.dumps(evaluations, indent=2))

    # Clean up Discriminator to free VRAM
    del discriminator_base
    del discriminator_model
    clear_gpu_memory()
    print("--- [Discriminator] Model unloaded from GPU.")

    # --- 3. CREATE FEEDBACK FOR NEXT LOOP ---
    avg_plausibility = (total_plausibility / len(evaluations)) if evaluations else 0
    feedback_from_discriminator = f"The previous plan's average plausibility score was {avg_plausibility:.2f}. The critique was: {json.dumps(evaluations)}. Focus on improving the plausibility of low-scoring steps."

print(f"\n{'=' * 20} ADVERSARIAL LOOP COMPLETE {'=' * 20}")
