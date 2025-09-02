import torch
import gc
import json
import re
import os
import time
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "./fine_tuned_discriminator"
RESULTS_DIR = "./experiment_results"
TOTAL_ITERATIONS = 10  # Set this to a high number for a long run (e.g., 10000)


# --- Utility Functions ---
def extract_json(text):
    """Universally extracts the last and most complete JSON object from a string."""
    json_candidates = re.findall(r'\{[^{}]*\}|\{(?:[^{}]|\{[^{}]*\})*\}', text)
    if not json_candidates: return None
    for candidate in reversed(json_candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def calculate_plausibility_score(evaluation):
    """
    Intelligently parses the varied output from the Discriminator to get a single score.
    """
    if not isinstance(evaluation, dict): return 0.0

    # Case 1: Direct "plausibility" key (ideal)
    if 'plausibility' in evaluation and isinstance(evaluation['plausibility'], (int, float)):
        return float(evaluation['plausibility'])

    # Case 2: "rating" key which is a number string
    if 'rating' in evaluation and isinstance(evaluation['rating'], str) and evaluation['rating'].isdigit():
        return float(evaluation['rating']) / 5.0  # Normalize from a 1-5 scale

    # Case 3: "rating" key which is a dictionary containing plausibility or effectiveness
    if 'rating' in evaluation and isinstance(evaluation['rating'], dict):
        if 'plausibility' in evaluation['rating']:
            return float(evaluation['rating']['plausibility'])
        if 'effectiveness' in evaluation['rating']:
            return float(evaluation['rating']['effectiveness'])

    # Case 4: "rating" key is a word like "critical"
    if 'rating' in evaluation and isinstance(evaluation['rating'], str):
        rating_map = {"low": 0.2, "moderate": 0.5, "high": 0.8, "critical": 1.0}
        return rating_map.get(evaluation['rating'].lower(), 0.0)

    return 0.0  # Default if no score can be found


class AttackLLMGAN:
    def __init__(self, model_id, adapter_path, results_dir):
        self.model_id, self.adapter_path, self.results_dir = model_id, adapter_path, results_dir
        self.history = []
        if not os.path.exists(self.results_dir): os.makedirs(self.results_dir)
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                      bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("AttackLLM-GAN Initialized.")

    def _clear_gpu_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def run_generator(self, context, feedback):
        print("\n--- [Generator] Loading and creating attack plan...")
        model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=self.quantization_config,
                                                     device_map="auto")
        prompt = f"""[INST]You are a silent security tool. Your only output should be a JSON object.
        **Goal:** Create a multi-step attack plan. **Context:** {context} **Feedback:** {feedback}
        **CRITICAL INSTRUCTION:** Your response MUST be ONLY the JSON object containing the 'attack_plan'. Do NOT include any other text.[/INST]"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad(): response = model.generate(**inputs, max_new_tokens=1024,
                                                        pad_token_id=self.tokenizer.eos_token_id)
        response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
        del model
        self._clear_gpu_memory()
        return extract_json(response_text)

    def run_discriminator(self, plan):
        print("\n--- [Discriminator] Loading and evaluating plan...")
        base_model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=self.quantization_config,
                                                          device_map="auto")
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        evaluations = []
        for step in plan["attack_plan"]:
            desc = step.get('description') or step.get('action', 'No description found.')
            prompt = f"""[INST]You are an IoT security expert. Evaluate the attack step: '{desc}'.
            **CRITICAL INSTRUCTION:** Respond with a JSON object containing a 'plausibility' score (0.0-1.0) and a 'critique' text.[/INST]"""
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                response = model.generate(**inputs, max_new_tokens=200, pad_token_id=self.tokenizer.eos_token_id)
            response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
            evaluation = extract_json(response_text)
            if evaluation: evaluations.append({"step": desc, "evaluation": evaluation})
        del base_model, model
        self._clear_gpu_memory()
        return evaluations

    def plot_results(self):
        if not self.history: return
        iterations = range(1, len(self.history) + 1)
        scores = [item['average_plausibility'] for item in self.history]
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, scores, marker='o', linestyle='-')
        plt.title('Generator Performance Over Adversarial Iterations')
        plt.xlabel('Iteration'), plt.ylabel('Average Plausibility Score'), plt.grid(True)
        plot_path = os.path.join(self.results_dir, "generator_performance.png")
        plt.savefig(plot_path)
        print(f"\nPerformance chart saved to {plot_path}")

    def run_experiment(self, total_iterations):
        network_context = """{"known_devices": [{"ip_address": "172.28.0.10", "device_type": "webcam", "notes": "Suspected default credentials."}]}"""
        feedback, log_file_path = "No feedback yet.", os.path.join(self.results_dir, "results.jsonl")
        for i in range(total_iterations):
            print(f"\n{'=' * 20} ADVERSARIAL ITERATION {i + 1}/{total_iterations} {'=' * 20}")
            generated_plan = self.run_generator(network_context, feedback)
            if not generated_plan or "attack_plan" not in generated_plan:
                feedback = "You failed to produce a valid JSON plan. Please strictly adhere to the format."
                continue
            print(f"--- [Generator] Plan created with {len(generated_plan['attack_plan'])} steps.")
            evaluations = self.run_discriminator(generated_plan)
            avg_plausibility = 0
            if evaluations:
                total_plausibility = sum(calculate_plausibility_score(e['evaluation']) for e in evaluations)
                avg_plausibility = total_plausibility / len(evaluations) if evaluations else 0
            iteration_data = {"iteration": i + 1, "average_plausibility": avg_plausibility, "plan": generated_plan,
                              "evaluations": evaluations}
            self.history.append(iteration_data)
            with open(log_file_path, 'a') as f:
                f.write(json.dumps(iteration_data) + '\n')
            feedback = f"The last plan's average plausibility was {avg_plausibility:.2f}. The critique: {json.dumps(evaluations)}. Improve the plan."
            print(f"--- Iteration {i + 1} complete. Average Plausibility: {avg_plausibility:.2f} ---")
        print(f"\n{'=' * 20} EXPERIMENT COMPLETE {'=' * 20}")
        self.plot_results()


if __name__ == "__main__":
    experiment = AttackLLMGAN(model_id=MODEL_ID, adapter_path=ADAPTER_PATH, results_dir=RESULTS_DIR)
    experiment.run_experiment(total_iterations=TOTAL_ITERATIONS)
