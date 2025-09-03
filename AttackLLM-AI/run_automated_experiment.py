import torch
import gc
import json
import re
import os
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "./fine_tuned_discriminator"
BASE_RESULTS_DIR = "./experiment_results"
TOTAL_ITERATIONS = 100


# --- Utility Functions ---
def extract_json(text):
    json_candidates = re.findall(r'\{[^{}]*\}|\{(?:[^{}]|\{[^{}]*\})*\}', text)
    if not json_candidates: return None
    for candidate in reversed(json_candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def calculate_metric_score(evaluation, key):
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

        # **NEW: Diversity tracking**
        self.plan_history = []  # Store raw plan text for embedding analysis
        self.diversity_metrics = []  # Track diversity over time
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        os.makedirs(self.results_dir, exist_ok=True)
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                      bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"AttackLLM-GAN Research Framework Initialized with Diversity Tracking")

    def _clear_gpu_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def run_generator(self, context, feedback):
        print("\n--- [Generator] Loading and creating attack plan...")
        model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=self.quantization_config,
                                                     device_map="auto")

        # **NEW: Add diversity pressure to the prompt**
        novelty_instruction = ""
        if len(self.plan_history) > 5:
            novelty_instruction = " **IMPORTANT: Create a novel attack plan that uses different techniques and approaches than previous attempts.**"

        prompt = f"""[INST]You are a silent security tool. Your only output should be a JSON object.
        **Goal:** Create a multi-step attack plan. **Context:** {context} **Feedback:** {feedback}{novelty_instruction}
        **CRITICAL INSTRUCTION:** Your response MUST be ONLY the JSON object containing the 'attack_plan'.[/INST]"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad(): response = model.generate(**inputs, max_new_tokens=1024,
                                                        pad_token_id=self.tokenizer.eos_token_id, do_sample=True,
                                                        temperature=0.8, top_p=0.9)
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

    def _calculate_diversity_metrics(self, plan, iteration):
        """Calculate multiple diversity metrics for mode collapse detection"""

        # Convert plan to text for embedding analysis
        plan_text = ' '.join([step.get('description', step.get('action', '')) for step in plan["attack_plan"]])
        self.plan_history.append(plan_text)

        # Calculate semantic similarity to previous plan
        cosine_similarity = 0.0
        if len(self.plan_history) > 1:
            emb_current = self.embedding_model.encode(self.plan_history[-1], convert_to_tensor=True)
            emb_previous = self.embedding_model.encode(self.plan_history[-2], convert_to_tensor=True)
            cosine_similarity = util.pytorch_cos_sim(emb_current, emb_previous).item()

        # Calculate rolling average similarity (mode collapse indicator)
        rolling_similarity = 0.0
        if len(self.plan_history) >= 5:
            recent_embeddings = [self.embedding_model.encode(text, convert_to_tensor=True) for text in
                                 self.plan_history[-5:]]
            similarities = []
            for i in range(len(recent_embeddings) - 1):
                sim = util.pytorch_cos_sim(recent_embeddings[i], recent_embeddings[i + 1]).item()
                similarities.append(sim)
            rolling_similarity = np.mean(similarities)

        # Extract technique diversity
        technique_ids = [step.get('technique_id', f'T{i}') for i, step in enumerate(plan["attack_plan"])]
        unique_techniques = len(set(technique_ids))

        # Calculate plan complexity metrics
        plan_length = len(plan["attack_plan"])
        avg_step_length = np.mean(
            [len(step.get('description', step.get('action', '')).split()) for step in plan["attack_plan"]])

        diversity_data = {
            "iteration": iteration,
            "cosine_similarity": cosine_similarity,
            "rolling_similarity": rolling_similarity,
            "unique_techniques": unique_techniques,
            "plan_length": plan_length,
            "avg_step_complexity": avg_step_length,
            "diversity_score": 1.0 - rolling_similarity  # Higher is more diverse
        }

        self.diversity_metrics.append(diversity_data)
        return diversity_data

    def _detect_mode_collapse(self):
        """Detect if the model has collapsed to a single mode"""
        if len(self.diversity_metrics) < 10:
            return False, "Insufficient data"

        recent_metrics = self.diversity_metrics[-10:]
        avg_rolling_similarity = np.mean([m["rolling_similarity"] for m in recent_metrics])
        technique_variance = np.var([m["unique_techniques"] for m in recent_metrics])
        length_variance = np.var([m["plan_length"] for m in recent_metrics])

        # Mode collapse indicators
        high_similarity = avg_rolling_similarity > 0.85
        low_technique_variance = technique_variance < 0.5
        low_length_variance = length_variance < 0.5

        collapse_detected = high_similarity and (low_technique_variance or low_length_variance)

        diagnosis = f"Avg Similarity: {avg_rolling_similarity:.3f}, Tech Variance: {technique_variance:.3f}, Length Variance: {length_variance:.3f}"

        return collapse_detected, diagnosis

    def save_and_plot_results(self):
        if not self.history: return
        successful_iterations = [item for item in self.history if item.get("generated_plan") and item.get("metrics")]
        if not successful_iterations: return

        iterations = [item['iteration'] for item in successful_iterations]
        metrics = [item['metrics'] for item in successful_iterations]

        # **NEW: Enhanced plotting with diversity metrics**
        fig, axs = plt.subplots(4, 1, figsize=(15, 24), sharex=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(f'AttackLLM-GAN Research Analysis ({timestamp})', fontsize=16)

        # Plot 1: Quality Metrics
        axs[0].plot(iterations, [m['avg_plausibility'] for m in metrics], marker='o', linestyle='-',
                    label='Avg Plausibility')
        axs[0].plot(iterations, [m['avg_stealth'] for m in metrics], marker='x', linestyle='--', label='Avg Stealth')
        axs[0].plot(iterations, [m['avg_impact'] for m in metrics], marker='^', linestyle=':', label='Avg Impact')
        axs[0].set_ylabel('Average Score (0.0 - 1.0)'), axs[0].set_title('Plan Quality Metrics')
        axs[0].legend(), axs[0].grid(True)

        # Plot 2: **NEW** Diversity Analysis
        if self.diversity_metrics:
            div_iterations = [d['iteration'] for d in self.diversity_metrics]
            axs[1].plot(div_iterations, [d['diversity_score'] for d in self.diversity_metrics], marker='o',
                        color='green', label='Diversity Score')
            axs[1].plot(div_iterations, [d['cosine_similarity'] for d in self.diversity_metrics], marker='s',
                        color='red', label='Similarity to Previous')
            axs[1].axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='Mode Collapse Threshold')
            axs[1].set_ylabel('Diversity Score'), axs[1].set_title('Mode Collapse Analysis')
            axs[1].legend(), axs[1].grid(True)

        # Plot 3: **NEW** Technique and Complexity Diversity
        if self.diversity_metrics:
            axs[2].bar(div_iterations, [d['unique_techniques'] for d in self.diversity_metrics], alpha=0.6,
                       color='blue', label='Unique Techniques')
            ax2_twin = axs[2].twinx()
            ax2_twin.plot(div_iterations, [d['plan_length'] for d in self.diversity_metrics], color='orange',
                          marker='d', label='Plan Length')
            axs[2].set_ylabel('Unique Techniques'), ax2_twin.set_ylabel('Plan Length')
            axs[2].set_title('Technique Diversity and Plan Complexity')
            axs[2].legend(loc='upper left'), ax2_twin.legend(loc='upper right'), axs[2].grid(True)

        # Plot 4: **NEW** Rolling Metrics for Mode Collapse Detection
        if len(self.diversity_metrics) >= 5:
            rolling_window = 5
            rolling_diversity = []
            rolling_iterations = []
            for i in range(rolling_window - 1, len(self.diversity_metrics)):
                window_data = self.diversity_metrics[i - rolling_window + 1:i + 1]
                avg_diversity = np.mean([d['diversity_score'] for d in window_data])
                rolling_diversity.append(avg_diversity)
                rolling_iterations.append(window_data[-1]['iteration'])

            axs[3].plot(rolling_iterations, rolling_diversity, marker='o', color='purple', linewidth=2)
            axs[3].axhline(y=0.15, color='red', linestyle='--', label='Collapse Warning')
            axs[3].set_ylabel('Rolling Diversity Score'), axs[3].set_title(
                f'{rolling_window}-Iteration Rolling Average Diversity')
            axs[3].legend(), axs[3].grid(True)

        plt.xlabel('Adversarial Iteration')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plot_path = os.path.join(self.results_dir, f"research_analysis_{time.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Research analysis chart saved to {plot_path}")

    def print_summary(self):
        print(f"\n{'=' * 20} RESEARCH SUMMARY {'=' * 20}")
        if not self.history: return

        successful_iterations = [item for item in self.history if item.get("generated_plan") and item.get("metrics")]
        total_runs, success_rate = len(self.history), len(successful_iterations) / len(
            self.history) if self.history else 0

        print(f"Results saved in: {self.results_dir}")
        print(f"Total Iterations Run: {total_runs}, Success Rate: {success_rate:.2%}")

        if not successful_iterations: return

        metrics = [item['metrics'] for item in successful_iterations]
        print("\n--- Performance Averages ---")
        print(f"  Average Plan Plausibility: {np.mean([m['avg_plausibility'] for m in metrics]):.3f}")
        print(f"  Average Plan Stealth:      {np.mean([m['avg_stealth'] for m in metrics]):.3f}")
        print(f"  Average Plan Impact:       {np.mean([m['avg_impact'] for m in metrics]):.3f}")

        # **NEW: Diversity Analysis Summary**
        if self.diversity_metrics:
            print("\n--- Diversity Analysis ---")
            avg_diversity = np.mean([d['diversity_score'] for d in self.diversity_metrics])
            avg_unique_techniques = np.mean([d['unique_techniques'] for d in self.diversity_metrics])
            plan_length_variance = np.var([d['plan_length'] for d in self.diversity_metrics])

            print(f"  Average Diversity Score:   {avg_diversity:.3f} (higher = more diverse)")
            print(f"  Average Unique Techniques: {avg_unique_techniques:.2f}")
            print(f"  Plan Length Variance:      {plan_length_variance:.3f}")

            # Mode collapse detection
            collapse_detected, diagnosis = self._detect_mode_collapse()
            print(f"\n--- Mode Collapse Analysis ---")
            print(f"  Mode Collapse Detected: {'YES - NEEDS ATTENTION' if collapse_detected else 'No'}")
            print(f"  Diagnosis: {diagnosis}")

            if collapse_detected:
                print("  ⚠️  RECOMMENDATION: Your model may have converged to repetitive outputs.")
                print(
                    "     Consider: 1) Adding more diversity pressure, 2) Adjusting temperature, 3) Curriculum learning")

        best_iteration = max(successful_iterations, key=lambda x: x['metrics']['avg_plausibility'])
        print(
            f"\n--- Best Iteration: #{best_iteration['iteration']} (Plausibility: {best_iteration['metrics']['avg_plausibility']:.3f}) ---")
        print(f"{'=' * 62}")

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
                # **NEW: Randomize feedback for failed attempts to encourage exploration**
                failure_responses = [
                    "You failed to produce a valid JSON plan. Try a completely different approach.",
                    "Invalid format. Explore novel attack vectors and be more creative.",
                    "Plan generation failed. Consider unconventional techniques and strategies."
                ]
                feedback = np.random.choice(failure_responses)
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

                # **NEW: Calculate diversity metrics**
                diversity_data = self._calculate_diversity_metrics(generated_plan, i + 1)

                iteration_data["evaluations"] = evaluations
                iteration_data["metrics"] = {
                    "avg_plausibility": avg_plausibility, "avg_stealth": avg_stealth, "avg_impact": avg_impact,
                    "plan_length": num_steps, "technique_diversity": unique_ttps
                }
                iteration_data["diversity"] = diversity_data

                # **NEW: Adaptive feedback based on diversity**
                if diversity_data["diversity_score"] < 0.2:  # Low diversity
                    feedback = f"Previous plan similarity: {diversity_data['cosine_similarity']:.2f}. CREATE A COMPLETELY DIFFERENT ATTACK STRATEGY. Avg plausibility was {avg_plausibility:.2f}. Explore new techniques and approaches."
                else:
                    feedback = f"The last plan's avg plausibility was {avg_plausibility:.2f}. Critique: {json.dumps(evaluations)}. Improve the plan."

                print(
                    f"--- Iteration {i + 1} complete. Plausibility: {avg_plausibility:.2f}, Diversity: {diversity_data['diversity_score']:.2f} ---")

            self.history.append(iteration_data)
            with open(log_file_path, 'a') as f:
                f.write(json.dumps(iteration_data) + '\n')

        self.save_and_plot_results()
        self.print_summary()


if __name__ == "__main__":
    experiment = AttackLLMGAN(model_id=MODEL_ID, adapter_path=ADAPTER_PATH, results_dir=os.path.join(BASE_RESULTS_DIR,
                                                                                                     f"research_run_{time.strftime('%Y%m%d_%H%M%S')}"))
    experiment.run_experiment(total_iterations=TOTAL_ITERATIONS)
