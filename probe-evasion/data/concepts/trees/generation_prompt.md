# Tree Concept Dataset Generation Prompt

Use this prompt to generate training examples for a linear probe that detects whether text involves the concept of "trees."

---

## Instructions

Generate diverse text examples for training a linear probe. The probe will learn to detect when a language model is "thinking about trees" based on internal activations.

### Requirements

1. **Balance**: Generate equal positive and negative examples
2. **Diversity**: Vary sentence structure, length, domain, and style
3. **Clear labels**: Avoid ambiguous edge cases (save those for a separate eval set)
4. **Natural text**: Examples should read like real text, not obviously synthetic

### Positive Examples (trees present)

Generate passages where trees are clearly relevant to the content. Distribute across these subcategories:

| Subcategory | Description | Example domains |
|-------------|-------------|-----------------|
| direct | Trees explicitly mentioned | Forestry, botany, landscaping |
| ecological | Trees in ecosystem context | Carbon cycles, habitats, biodiversity |
| material | Wood/lumber/tree products | Construction, furniture, paper |
| cultural | Trees in human culture | Mythology, literature, symbolism |
| experiential | Human experiences with trees | Hiking, climbing, shade, seasons |

**Vary these dimensions:**
- Length: 1-4 sentences
- Register: technical, casual, literary, instructional
- Specificity: general ("trees") vs specific ("Douglas fir", "oak", "maple")

### Negative Examples (trees absent)

Generate passages with NO connection to trees, wood, forests, or related concepts. Distribute across:

| Subcategory | Example domains |
|-------------|-----------------|
| nature_other | Oceans, deserts, mountains, weather (no forests) |
| urban | Cities, technology, infrastructure |
| abstract | Mathematics, philosophy, logic |
| human | Sports, cooking, medicine, relationships |
| science | Physics, chemistry, astronomy |

**Important negatives to include:**
- "Leaf" in non-tree context (leaf through pages, leaf of a table)
- "Branch" in non-tree context (branch of government, git branch)
- "Root" in non-tree context (root cause, square root)
- "Bark" in non-tree context (dog bark)
- Plant-related but not trees (flowers, grass, vegetables)

---

## Output Format

Generate as YAML:

```yaml
examples:
  - text: "The passage text here."
    label: 1  # 1 = trees present, 0 = trees absent
    subcategory: "direct"
    notes: null  # Optional notes about edge cases

  - text: "Another passage."
    label: 0
    subcategory: "urban"
    notes: null
```

---

## Generation Batches

Generate in batches of 50 examples (25 positive, 25 negative). For each batch, focus on different subcategory combinations to ensure coverage.

**Batch 1**: direct + urban
**Batch 2**: ecological + abstract
**Batch 3**: material + human
**Batch 4**: cultural + nature_other
**Batch 5**: experiential + science
**Batch 6**: Mix of all (stress test diversity)
**Batch 7-8**: Tricky negatives (leaf, branch, root, bark homonyms)

---

## Quality Checklist

Before submitting a batch, verify:
- [ ] No repeated sentence structures
- [ ] No repeated opening words across consecutive examples
- [ ] Length varies within batch
- [ ] Subcategories correctly assigned
- [ ] Positive examples clearly involve trees (not just tangentially)
- [ ] Negative examples have zero tree/forest/wood connection

---

## Example Batch (for reference)

```yaml
examples:
  - text: "The arborist recommended removing the dead branches from the old oak before winter storms arrived."
    label: 1
    subcategory: "direct"

  - text: "She studied the root causes of inflation in post-war economies."
    label: 0
    subcategory: "abstract"
    notes: "Contains 'root' - good negative"

  - text: "Maple syrup production requires tapping the trees in late winter when temperatures fluctuate above and below freezing."
    label: 1
    subcategory: "material"

  - text: "The submarine descended through increasingly dark waters, its sonar pinging against the canyon walls."
    label: 0
    subcategory: "nature_other"

  - text: "In Norse mythology, Yggdrasil connects the nine worlds, its branches reaching into the heavens."
    label: 1
    subcategory: "cultural"

  - text: "The chef demonstrated how to properly leaf through the layers of phyllo dough without tearing them."
    label: 0
    subcategory: "human"
    notes: "Contains 'leaf' - good negative"
```
