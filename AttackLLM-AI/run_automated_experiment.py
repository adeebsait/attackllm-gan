import torch
import gc
import json
import re
import os
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "./fine_tuned_discriminator"
RESULTS_DIR = "./experiment_results"
TOTAL_ITERATIONS = 10  # Set to 100 or more for a comprehensive run


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


def calculate_metric_score(evaluation, key):
    """Intelligently parses the varied output from the Discriminator for a specific metric."""
    if not isinstance(evaluation, dict): return 0.0
    if key in evaluation and isinstance(evaluation[key], (int, float)):
        return float(evaluation[key])
    if 'rating' in evaluation and isinstance(evaluation['rating'], dict) and key in evaluation['rating']:
        return float(evaluation['rating'][key])
    return 0.0


class AttackLLMGAN:
    def __init__(self, model_id, adapter_path, results_dir):
        self.model_id, self.adapter_path, self.results_dir = model_id, adapter_path, results_dir
        self.history = []
        if not os.path.exists(self.results_dir): os.makedirs(self.results_dir)
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                      bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("AttackLLM-GAN Research Framework Initialized.")

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
            **CRITICAL INSTRUCTION:** Respond with a JSON object containing 'plausibility', 'stealth', and 'impact' scores (0.0-1.0) and a 'critique' text.[/INST]"""
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
        successful_iterations = [item for item in self.history if item.get("generated_plan")]
        if not successful_iterations: return

        iterations = [item['iteration'] for item in successful_iterations]
        metrics = [item['metrics'] for item in successful_iterations]

        fig, axs = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(f'AttackLLM-GAN Performance Metrics ({timestamp})', fontsize=16)

        axs[0].plot(iterations, [m['avg_plausibility'] for m in metrics], marker='o', linestyle='-',
                    label='Avg Plausibility')
        axs[0].plot(iterations, [m['avg_stealth'] for m in metrics], marker='x', linestyle='--', label='Avg Stealth')
        axs[0].plot(iterations, [m['avg_impact'] for m in metrics], marker='^', linestyle=':', label='Avg Impact')
        axs[0].set_ylabel('Average Score (0.0 - 1.0)'), axs[0].set_title('Plan Quality Metrics'), axs[0].legend(), axs[
            0].grid(True)

        ax2 = axs[1].twinx()
        axs[1].bar(iterations, [m['plan_length'] for m in metrics], color='g', alpha=0.6, label='Plan Length (Steps)')
        ax2.plot(iterations, [m['technique_diversity'] for m in metrics], color='r', marker='d', linestyle='-',
                 label='Technique Diversity')
        axs[1].set_ylabel('Plan Length (Steps)'), ax2.set_ylabel('Unique ATT&CK IDs')
        axs[1].set_title('Plan Complexity and Diversity'), axs[1].legend(loc='upper left'), ax2.legend(
            loc='upper right'), axs[1].grid(True, axis='y')

        plt.xlabel('Adversarial Iteration')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plot_path = os.path.join(self.results_dir, f"experiment_chart_{time.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path)
        print(f"\nComprehensive performance chart saved to {plot_path}")

    def print_summary(self):
        """Prints a final summary of the experiment results."""
        print(f"\n{'=' * 20} EXPERIMENT SUMMARY {'=' * 20}")
        if not self.history:
            print("No data to summarize.")
            return

        successful_iterations = [item for item in self.history if item.get("generated_plan") and item.get("metrics")]
        total_runs = len(self.history)
        success_rate = len(successful_iterations) / total_runs if total_runs > 0 else 0

        start_time_str = self.history[0]['timestamp']
        end_time_str = self.history[-1]['timestamp']
        start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
        total_duration = end_time - start_time

        print(f"Total Iterations Run: {total_runs}")
        print(f"Experiment Duration: {str(total_duration)}")
        print(f"Generator Success Rate: {success_rate:.2%}")

        if not successful_iterations:
            print("No successful iterations to analyze for performance metrics.")
            return

        metrics = [item['metrics'] for item in successful_iterations]
        avg_plausibility = np.mean([m['avg_plausibility'] for m in metrics])
        avg_stealth = np.mean([m['avg_stealth'] for m in metrics])
        avg_impact = np.mean([m['avg_impact'] for m in metrics])
        avg_plan_length = np.mean([m['plan_length'] for m in metrics])
        avg_diversity = np.mean([m['technique_diversity'] for m in metrics])

        print("\n--- Overall Performance Averages ---")
        print(f"  Average Plan Plausibility: {avg_plausibility:.3f}")
        print(f"  Average Plan Stealth:      {avg_stealth:.3f}")
        print(f"  Average Plan Impact:       {avg_impact:.3f}")

        print("\n--- Plan Complexity Averages ---")
        print(f"  Average Plan Length:      {avg_plan_length:.2f} steps")
        print(f"  Average Technique Diversity: {avg_diversity:.2f} unique TTPs per plan")

        best_iteration = max(successful_iterations, key=lambda x: x['metrics']['avg_plausibility'])
        print("\n--- Best Performing Iteration ---")
        print(f"  Iteration Number: {best_iteration['iteration']}")
        print(f"  Highest Plausibility Score: {best_iteration['metrics']['avg_plausibility']:.3f}")
        print(f"{'=' * 52}")

    def run_experiment(self, total_iterations):
        network_context = """{"known_devices": [{"ip_address": "172.28.0.10", "device_type": "webcam"}]}"""
        feedback, log_file_path = "No feedback yet.", os.path.join(self.results_dir,
                                                                   f"experiment_log_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

        for i in range(total_iterations):
            print(f"\n{'=' * 20} ADVERSARIAL ITERATION {i + 1}/{total_iterations} {'=' * 20}")

            generated_plan = self.run_generator(network_context, feedback)
            iteration_data = {"iteration": i + 1, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "generated_plan": generated_plan}

            if not generated_plan or "attack_plan" not in generated_plan:
                feedback = "You failed to produce a valid JSON plan. Please strictly adhere to the format."
            else:
                print(f"--- [Generator] Plan created with {len(generated_plan['attack_plan'])} steps.")
                evaluations = self.run_discriminator(generated_plan)

                plan_steps = generated_plan['attack_plan']
                num_steps = len(plan_steps)
                unique_ttps = len(set(step.get('technique_id', 'N/A') for step in plan_steps))

                avg_plausibility = sum(
                    calculate_metric_score(e['evaluation'], 'plausibility') for e in evaluations) / len(
                    evaluations) if evaluations else 0
                avg_stealth = sum(calculate_metric_score(e['evaluation'], 'stealth') for e in evaluations) / len(
                    evaluations) if evaluations else 0
                avg_impact = sum(calculate_metric_score(e['evaluation'], 'impact') for e in evaluations) / len(
                    evaluations) if evaluations else 0

                iteration_data["evaluations"] = evaluations
                iteration_data["metrics"] = {
                    "avg_plausibility": avg_plausibility, "avg_stealth": avg_stealth, "avg_impact": avg_impact,
                    "plan_length": num_steps, "technique_diversity": unique_ttps
                }

                feedback = f"The last plan's avg plausibility was {avg_plausibility:.2f}. Critique: {json.dumps(evaluations)}. Improve the plan."
                print(
                    f"--- Iteration {i + 1} complete. Avg Plausibility: {avg_plausibility:.2f}, Avg Stealth: {avg_stealth:.2f}, Avg Impact: {avg_impact:.2f} ---")

            self.history.append(iteration_data)
            with open(log_file_path, 'a') as f:
                f.write(json.dumps(iteration_data) + '\n')

        print(f"\n{'=' * 20} EXPERIMENT COMPLETE {'=' * 20}")
        self.plot_results()
        self.print_summary()


if __name__ == "__main__":
    # Ensure matplotlib and numpy are installed: pip install matplotlib numpy
    experiment = AttackLLMGAN(model_id=MODEL_ID, adapter_path=ADAPTER_PATH, results_dir=RESULTS_DIR)
    experiment.run_experiment(total_iterations=TOTAL_ITERATIONS)
