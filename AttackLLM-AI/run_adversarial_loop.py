import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# --- Utility Functions (from previous scripts) ---

def extract_json(text):
    """Universally extracts a JSON object from a string, handling various markdown fences."""
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
                if open_braces == 0:
                    json_str = text[start_index: start_index + i + 1]
                    return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return None


# --- Model Loading ---

print("Loading all models... This may take a moment.")

# Common configuration
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 1. Load the Generator (Base Model)
generator_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config,
                                                       device_map="auto")
print("Generator model loaded.")

# 2. Load the Discriminator (Base Model + Fine-Tuned Adapter)
discriminator_base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config,
                                                                device_map="auto")
adapter_path = "./fine_tuned_discriminator"
discriminator_model = PeftModel.from_pretrained(discriminator_base_model, adapter_path)
print("Discriminator model loaded.")

# --- Adversarial Loop ---

# Initial cyber range context
network_context = """
{
  "known_devices": [{"ip_address": "172.28.0.10", "device_type": "webcam", "notes": "Suspected default credentials and command injection."}],
  "attacker_ip": "172.28.0.1"
}
"""
feedback_from_discriminator = "No feedback yet. This is the first attempt."
number_of_iterations = 3

for i in range(number_of_iterations):
    print(f"\n{'=' * 20} ADVERSARIAL ITERATION {i + 1}/{number_of_iterations} {'=' * 20}")

    # --- 1. GENERATOR'S TURN ---
    print("\n--- [Generator] Creating attack plan...")
    generator_prompt = f"""
    [INST]
    You are an expert automated red teamer. Your goal is to create a multi-step attack plan to gain initial access to a webcam and perform reconnaissance.

    **Network Context:**
    {network_context}

    **Feedback from Previous Attempt:**
    {feedback_from_discriminator}

    **Instructions:**
    - Analyze the context and feedback.
    - Create an improved, logical, step-by-step attack plan.
    - Your entire response must be ONLY a single, valid JSON object containing an "attack_plan" list.
    [/INST]
    """

    generator_input = tokenizer(generator_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        response = generator_model.generate(**generator_input, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    generated_plan = extract_json(response_text.split("[/INST]")[-1])

    if not generated_plan or "attack_plan" not in generated_plan:
        print("Generator failed to produce a valid plan. Skipping iteration.")
        feedback_from_discriminator = "You failed to produce a valid JSON attack plan. Please try again, ensuring your output is a single, complete JSON object."
        continue

    print(f"--- [Generator] Plan created with {len(generated_plan['attack_plan'])} steps.")
    print(json.dumps(generated_plan, indent=2))

    # --- 2. DISCRIMINATOR'S TURN ---
    print("\n--- [Discriminator] Evaluating plan...")
    evaluations = []
    total_plausibility = 0
    for step in generated_plan["attack_plan"]:
        discriminator_prompt = f"""
        [INST]
        You are an IoT security expert. Evaluate the following attack step. Provide your rating in a structured JSON format with 'plausibility', 'stealth', and 'impact' scores from 0.0 to 1.0, and a brief 'critique'.
        Attack Step: '{step.get('description', 'No description provided.')}'
        [/INST]
        """
        discriminator_input = tokenizer(discriminator_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            response = discriminator_model.generate(**discriminator_input, max_new_tokens=200,
                                                    pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        evaluation = extract_json(response_text.split("[/INST]")[-1])

        if evaluation:
            print(f"  - Step Evaluation: {evaluation}")
            evaluations.append({"step": step.get('description'), "evaluation": evaluation})
            total_plausibility += evaluation.get('plausibility', 0)
        else:
            print(f"  - Failed to evaluate step: {step.get('description')}")

    # --- 3. CREATE FEEDBACK FOR NEXT LOOP ---
    avg_plausibility = total_plausibility / len(generated_plan["attack_plan"]) if generated_plan["attack_plan"] else 0
    feedback_from_discriminator = f"The previous plan's average plausibility score was {avg_plausibility:.2f}. The detailed critique was: {json.dumps(evaluations, indent=2)}. Focus on improving the plausibility of the steps."

print(f"\n{'=' * 20} ADVERSARIAL LOOP COMPLETE {'=' * 20}")
