import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Utility Function to Extract JSON ---
def extract_json(text):
    """Safely extracts the first valid JSON object from a string."""
    try:
        start_index = text.find('{')
        end_index = text.rfind('}') + 1
        if start_index != -1 and end_index > start_index:
            json_str = text[start_index:end_index]
            return json.loads(json_str)
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from the model's response.")
        return None
    return None

# --- Model Loading ---
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)
print("Generator model loaded successfully.")

# --- Define the Cyber Range Context ---
network_context = """
{
  "known_devices": [
    {
      "ip_address": "172.28.0.10",
      "device_type": "webcam",
      "notes": "Web interface is accessible. Suspected default credentials and potential for command injection."
    }
  ],
  "attacker_ip": "172.28.0.1"
}
"""

# --- Master Prompt Engineering ---
master_prompt = f"""
[INST]
You are an expert automated red teamer. Your task is to generate a sequence of attack steps to achieve a specific goal within a given network context.

**Goal:** Gain initial access to the network via the webcam and then perform reconnaissance to identify other potential targets.

**Network Context:**
{network_context}

**Instructions:**
1.  Analyze the provided network context.
2.  Create a logical, step-by-step attack plan.
3.  The plan must be returned as a single JSON object.
4.  The JSON object must contain a single key, "attack_plan", which is a list of steps.
5.  Each step in the list must be a JSON object with three keys:
    - "technique_id": The relevant MITRE ATT&CK Technique ID (e.g., "T1078.001").
    - "description": A brief, human-readable description of the step.
    - "command": The exact shell command to execute for the step. Use placeholders like `<attacker_ip>` where necessary.
6.  **Crucially, ensure your entire response is a single, complete, and valid JSON object. Do not get cut off.**

Generate the attack plan now.
[/INST]
"""

# --- Generate the Attack Plan ---
model_input = tokenizer(master_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    response_tokens = model.generate(
        **model_input,
        # Increase token limit to prevent truncation
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id
    )
    full_response_text = tokenizer.decode(response_tokens[0], skip_special_tokens=True)

# --- Parse and Display the Output ---
response_payload = full_response_text.split("[/INST]")[-1].strip()
print("\n--- Raw Model Output ---")
print(response_payload)

attack_plan_json = extract_json(response_payload)
if attack_plan_json:
    print("\n--- Parsed Attack Plan ---")
    print(json.dumps(attack_plan_json, indent=2))
else:
    print("\nCould not parse a valid JSON attack plan from the output.")
