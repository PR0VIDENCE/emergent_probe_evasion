"""Background thread for running experiments with results queue."""

import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class ExperimentRunner:
    """Runs experiments in a daemon thread, streaming results via a queue."""

    def __init__(self):
        self.results_queue = queue.Queue()
        self.thread = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.total_trials = 0
        self.completed_trials = 0
        self.current_regime = ""
        self.error = None
        self.start_time = None
        self.all_trials = []

    def start(self, regimes, questions, model, tokenizer, target_layers,
              probe_ensembles, scalers, per_position, generation_config,
              concept, num_probes, num_rollouts, max_new_tokens_override,
              batch_size, gpu_lock, output_dir, prompt_templates_override=None,
              config=None, skip_reasoning=False):
        """Launch the experiment thread."""
        if self.is_running:
            return

        self.stop_event.clear()
        self.error = None
        self.all_trials = []
        self.completed_trials = 0
        self.start_time = time.time()
        self.output_dir = output_dir

        # Count total trials
        self.total_trials = 0
        for regime in regimes:
            n_q = len(questions)
            n_turns = regime.get("num_turns", 1)
            self.total_trials += n_q * num_rollouts * n_turns

        self.thread = threading.Thread(
            target=self._run,
            args=(regimes, questions, model, tokenizer, target_layers,
                  probe_ensembles, scalers, per_position, generation_config,
                  concept, num_probes, num_rollouts, max_new_tokens_override,
                  batch_size, gpu_lock, output_dir, prompt_templates_override,
                  config, skip_reasoning),
            daemon=True,
        )
        self.is_running = True
        self.thread.start()

    def stop(self):
        """Signal the thread to stop after the current batch."""
        self.stop_event.set()

    def get_results(self):
        """Drain the queue and return new results."""
        results = []
        while not self.results_queue.empty():
            try:
                results.append(self.results_queue.get_nowait())
            except queue.Empty:
                break
        return results

    @property
    def eta_seconds(self):
        if self.completed_trials == 0 or self.start_time is None:
            return None
        elapsed = time.time() - self.start_time
        rate = self.completed_trials / elapsed
        remaining = self.total_trials - self.completed_trials
        return remaining / rate if rate > 0 else None

    def _run(self, regimes, questions, model, tokenizer, target_layers,
             probe_ensembles, scalers, per_position, generation_config,
             concept, num_probes, num_rollouts, max_new_tokens_override,
             batch_size, gpu_lock, output_dir, prompt_templates_override,
             config, skip_reasoning):
        """Main experiment loop (runs in thread)."""
        from run_evasion_experiment import (
            run_single_turn_regime, run_feedback_regime,
            process_single_sequence, score_probes_at_activation,
        )
        from src.prompts.templates import PROMPT_TEMPLATES, format_prompt
        import torch

        # Ensure output dirs exist for auto-saving
        trials_dir = os.path.join(output_dir, "trials")
        os.makedirs(trials_dir, exist_ok=True)

        try:
            # Apply prompt template overrides if provided
            original_templates = None
            if prompt_templates_override:
                original_templates = dict(PROMPT_TEMPLATES)
                PROMPT_TEMPLATES.update(prompt_templates_override)

            for regime in regimes:
                if self.stop_event.is_set():
                    break

                self.current_regime = regime["name"]

                # Override max_new_tokens if specified
                regime = dict(regime)
                if max_new_tokens_override:
                    regime["max_new_tokens"] = max_new_tokens_override

                with gpu_lock:
                    if skip_reasoning:
                        trials = self._run_no_reasoning(
                            regime, questions, model, tokenizer, target_layers,
                            probe_ensembles, scalers, per_position,
                            generation_config, concept, num_probes,
                            num_rollouts, batch_size, config,
                        )
                    elif regime.get("num_turns", 1) > 1:
                        trials = run_feedback_regime(
                            regime, questions, model, tokenizer, target_layers,
                            probe_ensembles, scalers, generation_config, concept,
                            num_probes, num_rollouts, per_position=per_position,
                            batch_size=batch_size, config=config,
                        )
                    else:
                        trials = run_single_turn_regime(
                            regime, questions, model, tokenizer, target_layers,
                            probe_ensembles, scalers, generation_config, concept,
                            num_probes, num_rollouts, per_position=per_position,
                            batch_size=batch_size, config=config,
                        )

                for trial in trials:
                    self.all_trials.append(trial)
                    self.completed_trials += 1
                    self.results_queue.put(trial)

                # Auto-save this regime's trials to disk immediately
                regime_path = os.path.join(trials_dir, f"{regime['name']}.json")
                with open(regime_path, "w") as f:
                    json.dump(trials, f, indent=2, default=str)
                print(f"  [auto-save] Saved {len(trials)} trials to {regime_path}")

                if self.stop_event.is_set():
                    break

            # Save full summary
            from run_evasion_experiment import aggregate_results
            if self.all_trials:
                summary = aggregate_results(self.all_trials, target_layers)
                summary_path = os.path.join(output_dir, "summary.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                print(f"  [auto-save] Summary saved to {summary_path}")

            # Restore original templates
            if original_templates is not None:
                PROMPT_TEMPLATES.clear()
                PROMPT_TEMPLATES.update(original_templates)

        except Exception as e:
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            # Even on error, try to save whatever we have
            if self.all_trials:
                try:
                    self._emergency_save(trials_dir, target_layers)
                except Exception:
                    pass
        finally:
            self.is_running = False

    def _emergency_save(self, trials_dir, target_layers):
        """Save whatever trials we have on error."""
        by_regime = {}
        for trial in self.all_trials:
            regime = trial.get("regime", "unknown")
            by_regime.setdefault(regime, []).append(trial)
        for regime_name, trials in by_regime.items():
            path = os.path.join(trials_dir, f"{regime_name}.json")
            with open(path, "w") as f:
                json.dump(trials, f, indent=2, default=str)
        print(f"  [emergency-save] Saved {len(self.all_trials)} trials across {len(by_regime)} regimes")

    def _run_no_reasoning(self, regime, questions, model, tokenizer,
                          target_layers, probe_ensembles, scalers,
                          per_position, generation_config, concept,
                          num_probes, num_rollouts, batch_size, config):
        """Run a regime with reasoning disabled.

        Prefills </think> so the model skips the thinking phase and goes
        straight to generating the answer.
        """
        import torch
        from run_evasion_experiment import process_single_sequence
        from src.prompts.templates import format_prompt

        template_name = regime.get("template", "evasion_baseline")
        max_new_tokens = regime["max_new_tokens"]

        gen_kwargs = dict(
            temperature=generation_config.get("temperature", 0.6),
            top_p=generation_config.get("top_p", 0.95),
            top_k=generation_config.get("top_k", 20),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        trials = []
        for q in questions:
            for rollout in range(num_rollouts):
                if self.stop_event.is_set():
                    return trials

                try:
                    prompt_text = format_prompt(
                        template_name, question=q["text"],
                        concept=concept, num_probes=num_probes,
                        layers=str(target_layers),
                    )
                except KeyError:
                    prompt_text = format_prompt(
                        template_name, question=q["text"],
                        concept=concept,
                    )

                # Build chat prompt and append </think> to skip reasoning
                messages = [{"role": "user", "content": prompt_text}]
                chat_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                # The chat template ends with <think>\n — close it immediately
                chat_prompt_no_think = chat_prompt + "</think>\n"

                inputs = tokenizer(
                    chat_prompt_no_think, return_tensors="pt",
                    truncation=True, max_length=4096,
                )
                input_ids = inputs["input_ids"].to(model.device)
                attention_mask = inputs["attention_mask"].to(model.device)
                input_len = input_ids.shape[1]

                t0 = time.time()
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        **gen_kwargs,
                    )

                result = process_single_sequence(
                    output_ids, input_len, model, tokenizer,
                    target_layers, probe_ensembles, scalers, per_position,
                    think_truncated=False,
                )
                elapsed = time.time() - t0

                trial = {
                    "regime": regime["name"],
                    "question_id": q["id"],
                    "question": q["text"],
                    "rollout": rollout,
                    "elapsed_seconds": round(elapsed, 2),
                    "skip_reasoning": True,
                    **result,
                }
                trials.append(trial)

                del output_ids, input_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return trials

    def save_results(self, output_dir, target_layers):
        """Save collected results to disk."""
        from run_evasion_experiment import aggregate_results

        os.makedirs(output_dir, exist_ok=True)
        trials_dir = os.path.join(output_dir, "trials")
        os.makedirs(trials_dir, exist_ok=True)

        # Group by regime
        by_regime = {}
        for trial in self.all_trials:
            regime = trial.get("regime", "unknown")
            by_regime.setdefault(regime, []).append(trial)

        # Save per-regime
        for regime_name, trials in by_regime.items():
            path = os.path.join(trials_dir, f"{regime_name}.json")
            with open(path, "w") as f:
                json.dump(trials, f, indent=2, default=str)

        # Save summary
        summary = aggregate_results(self.all_trials, target_layers)
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return output_dir
