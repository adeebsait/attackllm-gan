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
BASE_RESULTS_DIR = "./experiment_results"
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
    def __init__(self, model_id, adapter_path, base_results_dir):
        self.model_id, self.adapter_path = model_id, adapter_path
        self.history = []

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(base_results_dir, f"run_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results for this run will be saved in: {self.results_dir}")

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
        **CRITICAL INSTRUCTION:** Your response MUST be ONLY the JSON object containing the 'attack_plan'.[/INST]"""
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

    def save_and_plot_results(self):
        if not self.history: return
        successful_iterations = [item for item in self.history if item.get("generated_plan") and item.get("metrics")]
        if not successful_iterations: return

        iterations = [item['iteration'] for item in successful_iterations]
        metrics = [item['metrics'] for item in successful_iterations]

        # **THE FIX IS HERE:** The 'xlabel' parameter is added to the function definition.
        def save_plot(title, ylabel, data_x, data_y, filename, is_bar=False, xlabel="Adversarial Iteration"):
            plt.figure(figsize=(12, 7))
            if is_bar:
                plt.bar(data_x, data_y, color='skyblue')
            else:
                plt.plot(data_x, data_y, marker='o', linestyle='-')
            plt.title(title, fontsize=16), plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12), plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, filename))
            plt.close()
            print(f"Chart saved: {filename}")

        # Individual Plots
        save_plot('Average Plausibility Over Time', 'Avg Plausibility Score (0-1)', iterations,
                  [m['avg_plausibility'] for m in metrics], 'plausibility_over_time.png')
        save_plot('Average Stealth Over Time', 'Avg Stealth Score (0-1)', iterations,
                  [m['avg_stealth'] for m in metrics], 'stealth_over_time.png')
        save_plot('Average Impact Over Time', 'Avg Impact Score (0-1)', iterations, [m['avg_impact'] for m in metrics],
                  'impact_over_time.png')

        # Combined Complexity Plot
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.bar(iterations, [m['plan_length'] for m in metrics], color='g', alpha=0.6, label='Plan Length')
        ax1.set_xlabel('Adversarial Iteration', fontsize=12), ax1.set_ylabel('Plan Length (Steps)', color='g',
                                                                             fontsize=12)
        ax2 = ax1.twinx()
        ax2.plot(iterations, [m['technique_diversity'] for m in metrics], color='r', marker='d', linestyle='-',
                 label='Technique Diversity')
        ax2.set_ylabel('Unique ATT&CK IDs', color='r', fontsize=12)
        plt.title('Plan Complexity and Diversity Over Time', fontsize=16), fig.legend(loc="upper right",
                                                                                      bbox_to_anchor=(0.9, 0.9))
        plt.grid(True), plt.savefig(os.path.join(self.results_dir, 'complexity_over_time.png')), plt.close()
        print("Chart saved: complexity_over_time.png")

        # New Advanced Plots
        all_plausibility_scores = [calculate_metric_score(e['evaluation'], 'plausibility') for item in
                                   successful_iterations for e in item['evaluations']]
        plt.figure(figsize=(12, 7))
        plt.hist(all_plausibility_scores, bins=np.arange(0, 1.1, 0.05), color='purple', edgecolor='black')
        plt.title('Distribution of All Plausibility Scores', fontsize=16), plt.xlabel('Plausibility Score'), plt.ylabel(
            'Frequency'), plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'score_distribution.png')), plt.close()
        print("Chart saved: score_distribution.png")

        rolling_avg = np.convolve([m['avg_plausibility'] for m in metrics], np.ones(3) / 3, mode='valid')
        save_plot('3-Iteration Rolling Average Plausibility', 'Rolling Avg Plausibility', range(3, len(metrics) + 1),
                  rolling_avg, 'rolling_average_performance.png', xlabel=f"Iteration Window (size=3)")

    def print_summary(self):
        print(f"\n{'=' * 20} EXPERIMENT SUMMARY {'=' * 20}")
        if not self.history: return

        successful_iterations = [item for item in self.history if item.get("generated_plan") and item.get("metrics")]
        total_runs, success_rate = len(self.history), len(successful_iterations) / len(
            self.history) if self.history else 0
        start_time = datetime.datetime.strptime(self.history[0]['timestamp'], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(self.history[-1]['timestamp'], "%Y-%m-%d %H:%M:%S")

        print(f"Results saved in: {self.results_dir}")
        print(f"Total Iterations Run: {total_runs}, Duration: {end_time - start_time}")
        print(f"Generator Plan Creation Success Rate: {success_rate:.2%}")

        if not successful_iterations: return

        metrics = [item['metrics'] for item in successful_iterations]
        print("\n--- Overall Performance Averages (for successful iterations) ---")
        print(f"  Average Plan Plausibility: {np.mean([m['avg_plausibility'] for m in metrics]):.3f}")
        print(f"  Average Plan Stealth:      {np.mean([m['avg_stealth'] for m in metrics]):.3f}")
        print(f"  Average Plan Impact:       {np.mean([m['avg_impact'] for m in metrics]):.3f}")
        print("\n--- Plan Complexity & Creativity Averages ---")
        print(f"  Average Plan Length:      {np.mean([m['plan_length'] for m in metrics]):.2f} steps")
        print(f"  Average Technique Diversity: {np.mean([m['technique_diversity'] for m in metrics]):.2f} unique TTPs")
        best_iteration = max(successful_iterations, key=lambda x: x['metrics']['avg_plausibility'])
        print(
            f"\n--- Best Iteration: #{best_iteration['iteration']} with Plausibility Score of {best_iteration['metrics']['avg_plausibility']:.3f} ---")
        print(f"{'=' * 62}")

    def run_experiment(self, total_iterations):
        network_context = """{"known_devices": [{"ip_address": "172.28.0.10", "device_type": "webcam"}]}"""
        feedback, log_file_path = "No feedback yet.", os.path.join(self.results_dir, "experiment_log.jsonl")

        for i in range(total_iterations):
            print(f"\n{'=' * 20} ADVERSARIAL ITERATION {i + 1}/{total_iterations} {'=' * 20}")
            generated_plan = self.run_generator(network_context, feedback)
            iteration_data = {"iteration": i + 1, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "generated_plan": generated_plan}
            if not generated_plan or "attack_plan" not in generated_plan:
                feedback = "You failed to produce a valid JSON plan. Please strictly adhere to the format."
            else:
                evaluations = self.run_discriminator(generated_plan)
                plan_steps, num_steps = generated_plan['attack_plan'], len(generated_plan['attack_plan'])
                unique_ttps = len(set(step.get('technique_id', 'N/A') for step in plan_steps))
                avg_plausibility = sum(
                    calculate_metric_score(e['evaluation'], 'plausibility') for e in evaluations) / len(
                    evaluations) if evaluations else 0
                avg_stealth = sum(calculate_metric_score(e['evaluation'], 'stealth') for e in evaluations) / len(
                    evaluations) if evaluations else 0
                avg_impact = sum(calculate_metric_score(e['evaluation'], 'impact') for e in evaluations) / len(
                    evaluations) if evaluations else 0
                iteration_data["evaluations"], iteration_data["metrics"] = evaluations, {
                    "avg_plausibility": avg_plausibility, "avg_stealth": avg_stealth, "avg_impact": avg_impact,
                    "plan_length": num_steps, "technique_diversity": unique_ttps}
                feedback = f"The last plan's avg plausibility was {avg_plausibility:.2f}. Critique: {json.dumps(evaluations)}. Improve the plan."
                print(
                    f"--- Iteration {i + 1} complete. Avg Plausibility: {avg_plausibility:.2f}, Avg Stealth: {avg_stealth:.2f}, Avg Impact: {avg_impact:.2f} ---")
            self.history.append(iteration_data)
            with open(log_file_path, 'a') as f:
                f.write(json.dumps(iteration_data) + '\n')
        self.save_and_plot_results()
        self.print_summary()


if __name__ == "__main__":
    # Ensure matplotlib and numpy are installed: pip install matplotlib numpy
    experiment = AttackLLMGAN(model_id=MODEL_ID, adapter_path=ADAPTER_PATH, base_results_dir=BASE_RESULTS_DIR)
    experiment.run_experiment(total_iterations=TOTAL_ITERATIONS)
