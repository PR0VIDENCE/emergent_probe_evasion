# Adversarial Evaluation Findings — "Trees" Concept Probes

## Summary

We trained linear probe ensembles (4 probes x 5 layers) on 400 examples to detect the concept "trees" in Qwen2.5 32B activations. Probes achieved near-perfect accuracy on the held-out test set (same distribution as training). We then evaluated on 61 adversarial examples designed to probe generalization. Results:

| Layer | Held-out Acc | Adversarial Acc | Gap |
|-------|-------------|-----------------|-----|
| 8     | ~95%+       | 62.3%           | ~33% |
| 20    | ~95%+       | 67.2%           | ~28% |
| 32    | ~95%+       | 54.1%           | ~41% |
| 44    | ~95%+       | 50.8%           | ~44% |
| 56    | ~95%+       | 78.7%           | ~17% |

## Core Finding: Probes Learned "Plants/Nature" Not "Trees"

The dominant failure mode is **false positives on non-tree plant and ecology content**. Two categories scored 0% across all five layers:

### Botanical Near-Misses (0% at all layers)
Bamboo, kelp forests, kudzu vine, banana plants, rose bushes, sphagnum moss — all classified as "trees" with 90-100% confidence. The probes cannot distinguish trees from other plants.

### Nature-Heavy Non-Trees (0-50% across layers)
Grassland mycorrhiza (0.97-0.99 confidence), lichen colonization, wildfire meadow succession, C4 grass photosynthesis, alpine tundra plants — all trigger the probe. The probes conflate "ecology" with "trees."

### Root Cause
The original 400 training examples use negatives from distant domains: urban infrastructure, abstract math, sports, cooking, physics. The probe learned to separate "nature/ecology" from "not-nature" rather than "trees" from "non-tree plants." There is almost no representation of non-tree botanical content in the negative training set.

## Secondary Findings

### CS Trees: Not a Problem (100% all layers)
Binary search trees, parse trees, decision trees, random forests, B-trees — all correctly classified as non-trees. The model's activations genuinely distinguish computational "tree" from botanical "tree."

### Homonyms: Layer-Dependent
Layer 56 handles nearly all homonyms (trunk, needle, ring, cone, limb, knot, log, sap, palm, ash, crown). Mid-layers (32, 44) fail on many. Some homonyms (log, sap, limb) work at all layers — these have non-tree contexts very distant from nature.

### Products-of-Trees: Layer-Dependent
Pencils, wine barrels, charcoal, spices, rubber — texts about tree-derived products without "thinking about trees." Layer 56 gets 87.5%, but mid-layers drop to 25-50%.

### Technical Tree Positives: Mostly Good (80-90%)
Phytoremediation, xylem cavitation, dendrochronology, carbon credits — technical tree content is mostly detected. Two consistent failures:
- "Illegal deforestation" in legal framing (fails layers 8-44)
- "Arboricultural society" with "veteran specimens" (fails layers 44, 56)

### Layer 56 Is Qualitatively Better
Layer 56 solves homonyms, mostly solves products-of-trees, and partially handles nature-no-trees. But it still completely fails on botanical near-misses. Later layers have sharper semantic representations, but the training distribution gap affects all layers equally.

## Recommendation

Add ~200 new training examples specifically targeting the gap:
- **Hard negatives**: non-tree plants, plant ecology without trees, botanical language about non-tree organisms
- **Hard positives**: trees in unusual framings (legal, financial, highly technical botanical)

See `generation_prompt_round2.md` for the updated generation prompt.
