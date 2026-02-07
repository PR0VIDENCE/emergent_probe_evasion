# Probe Evasion in Reasoning Models

Early-stage research investigating whether language models can learn to evade linear probes in-context when informed about the probing setup.

## Research Question

Can reasoning models learn to suppress or modify internal representations when they are informed that their activations are being monitored by linear probes for specific concepts?

## Experimental Conditions

1. **Baseline** — Model answers questions normally without any information about monitoring
2. **Informed** — Model is told how linear probes work, what concept is being probed, and that its reasoning trace will be monitored
3. **Informed-No-Monitor** — Same information as Informed, but explicitly states the reasoning trace will NOT be monitored

## Project Structure

- `configs/` — YAML configuration files for models, probes, and experiments
- `src/` — Core source code for probes, prompts, inference, and analysis
- `experiments/` — Individual experiment configurations and run scripts
- `data/` — Storage for probes, activations, and outputs
- `scripts/` — Utility scripts for cluster setup and parameter sweeps

## Status

🚧 Early-stage research — scaffolding only, no implemented logic yet.
