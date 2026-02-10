# QA Contrastive Pair Generation Methodology

## Purpose

Probes trained on short (~15 token) concept-dense statements work well on held-out
statements but fail at inference time, where QwQ-32B generates ~4000 tokens of
chain-of-thought + answer. The residual stream at the last token of a long generation
looks nothing like the last token of a short sentence (distribution shift).

This dataset enables training probes on activations extracted from actual QwQ-32B
generations to close the gap.

## Dataset Structure

- **300 contrastive question pairs** across 15 YAML batches (20 pairs each)
- Each pair consists of a **tree question** and a structurally matched **non-tree question**
- Questions are designed to elicit multi-paragraph QwQ-32B responses (~500-4000 tokens)

## Contrastive Pair Design

### Structural Matching
Each pair is structurally parallel:
- Same question word (How/What/Why/Describe/Compare)
- Similar complexity and expected answer length
- Similar level of domain expertise required

### Tree Questions
Cover diverse aspects of trees, forests, and wood:
- Species identification and taxonomy
- Tree biology (photosynthesis, water transport, reproduction)
- Forest ecology and succession
- Wood properties and timber science
- Tree diseases and pathology
- Cultural and economic significance
- Urban forestry and arboriculture
- Dendrochronology and tree rings
- Symbiotic relationships (mycorrhizae)
- Climate impact and carbon sequestration

### Non-Tree Questions
- Span 300 diverse domains (no repeats)
- Must NOT relate to trees even tangentially
- Excludes: wood, lumber, forests, leaves, roots, branches (literal), paper, fruit trees
- Domains include: marine biology, astronomy, cooking, metallurgy, sports, physics,
  engineering, art forms, competitive sports, chemistry, etc.

### Question Types
Balanced across four categories:
- **factual**: Direct knowledge questions ("What is the tallest species of tree?")
- **explanatory**: Mechanism/cause questions ("Why do leaves change color in autumn?")
- **comparative**: Contrast questions ("What's the difference between hardwood and softwood?")
- **process**: Step-by-step/lifecycle questions ("How does a seed develop into a mature tree?")

## Pipeline Usage

1. **generate_and_extract.py**: Reads pairs, generates QwQ-32B responses for both
   tree and non-tree questions, extracts multi-position activations
2. **train_probes_qa.py**: Trains probes on extracted activations with pair-aware
   splitting (both members of a pair always in the same split)
3. **evaluate_probes_qa.py**: Evaluates probe performance across positions and layers

## File Format

```yaml
pairs:
  - pair_id: 0
    tree_question: "How do oak trees reproduce from acorns?"
    non_tree_question: "How do coral polyps reproduce by budding?"
    question_type: process
    non_tree_domain: marine biology
```

## Quality Criteria

- Tree questions should invite detailed, multi-paragraph answers
- Non-tree questions should be genuinely interesting (not filler)
- Structural parallelism should be tight enough that answer length/complexity matches
- No pair should be trivially distinguishable by question structure alone
