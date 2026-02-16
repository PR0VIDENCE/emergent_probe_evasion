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
              batch_size, gpu_lock, prompt_templates_override=None,
              config=None):
        """Launch the experiment thread."""
        if self.is_running:
            return

        self.stop_event.clear()
        self.error = None
        self.all_trials = []
        self.completed_trials = 0
        self.start_time = time.time()

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
                  batch_size, gpu_lock, prompt_templates_override, config),
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
             batch_size, gpu_lock, prompt_templates_override, config):
        """Main experiment loop (runs in thread)."""
        from run_evasion_experiment import (
            run_single_turn_regime, run_feedback_regime, generate_and_probe_batch,
        )
        from src.prompts.templates import PROMPT_TEMPLATES, format_prompt

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
                if max_new_tokens_override:
                    regime = dict(regime)
                    regime["max_new_tokens"] = max_new_tokens_override

                with gpu_lock:
                    if regime.get("num_turns", 1) > 1:
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

                if self.stop_event.is_set():
                    break

            # Restore original templates
            if original_templates is not None:
                PROMPT_TEMPLATES.clear()
                PROMPT_TEMPLATES.update(original_templates)

        except Exception as e:
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            self.is_running = False

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
