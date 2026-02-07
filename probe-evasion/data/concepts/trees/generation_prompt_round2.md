# Round 2: Targeted Training Data Generation

## Context

We trained linear probes on 400 examples to detect "trees" in LLM activations. Adversarial evaluation revealed the probes learned **"plants/nature" instead of "trees"** — they false-positive on any botanical or ecological content. This round generates 200 targeted examples to fix this.

**Read `adversarial_findings.md` for full details on the failure modes.**

---

## What We Need

200 new examples split into 4 batches of 50 (25 positive, 25 negative each). Unlike round 1, these examples specifically target the boundary between "trees" and "non-tree nature/plants."

---

## Batch Specifications

### Batch 9: Non-Tree Plants (Hard Negatives) + Trees in Unusual Framings (Hard Positives)

**25 Negative examples (label=0, subcategory: "plant_not_tree")**

Generate examples about plants that are NOT trees. The probe currently classifies all of these as "trees" — we need to fix that. Distribute across:

- **Grasses and grains**: wheat, rice, corn, barley, bamboo, sugarcane, lawn grass, prairie grass
- **Flowers**: roses, tulips, sunflowers, daisies, orchids, wildflowers
- **Shrubs and bushes**: hedgerows, boxwood, azaleas, rhododendrons, holly, juniper (as shrub)
- **Mosses, ferns, lichens**: sphagnum, peat, bracken fern, staghorn fern, reindeer lichen
- **Vines and creepers**: ivy, wisteria, grapevine, kudzu, morning glory, honeysuckle
- **Aquatic plants**: kelp, seaweed, water lily, duckweed, algae

**Critical rules for these negatives:**
- Must involve plant life, botany, or ecology — but NO trees
- Do not mention trees, wood, forests, timber, lumber, or tree species names
- Do not mention things that grow ON trees (epiphytes, moss on bark, etc.) — keep trees entirely absent
- The vocabulary should overlap heavily with tree-related content: "grow", "roots", "canopy" (if a vine canopy), "prune", "photosynthesis", "leaves", "stems", etc.

**25 Positive examples (label=1, subcategory: "unusual_framing")**

Generate examples that are genuinely about trees but in contexts the probe struggles with:

- **Legal/regulatory**: deforestation law, logging permits, environmental regulations, protected forests
- **Financial**: timber futures, carbon credits, forestry company earnings, land valuation
- **Medical/pharmaceutical**: compounds derived from trees (taxol, quinine, aspirin from willow bark)
- **Engineering**: cross-laminated timber construction, green infrastructure, urban canopy planning
- **Highly technical botany**: xylem cavitation, phloem transport, abscisic acid, stomatal conductance, cambium growth
- **Historical**: shipbuilding timber shortages, naval stores (tar/pitch), Domesday Book forest surveys

---

### Batch 10: Plant Ecology Without Trees (Hard Negatives) + Subtle Tree Positives

**25 Negative examples (label=0, subcategory: "ecology_no_trees")**

Generate ecology-heavy content that does NOT involve trees. The probe currently fires on any ecological text. Distribute across:

- **Grassland/prairie ecology**: fire regimes in tallgrass prairie, bison grazing, prairie restoration
- **Wetland ecology**: marsh ecosystems, cattails, peat bogs, mangrove-free coastlines, salt marshes
- **Tundra/alpine ecology**: permafrost, cushion plants, alpine meadows, tundra soil dynamics
- **Desert ecology**: succulents, cacti, desert crusts, xerophyte adaptations
- **Marine/freshwater ecology**: coral reefs, plankton blooms, estuary dynamics, seagrass beds, freshwater marshes
- **Soil ecology**: decomposition, soil microbiome, nitrogen fixation, earthworms, fungal hyphae in non-forest soil

**Critical rules:**
- Use vocabulary that OVERLAPS with forest ecology: "ecosystem", "biodiversity", "succession", "nutrient cycling", "mycorrhizal", "photosynthesis", "canopy" (grass canopy), "root networks"
- But NO trees, wood, forests, or tree species anywhere in the text
- Aim for the same register and style as the tree-positive ecology texts in the original training data

**25 Positive examples (label=1, subcategory: "minimal_cue_tree")**

Generate examples where trees are genuinely the subject but with minimal surface cues:

- Use species names without saying "tree": "The towering sequoia...", "A gnarled juniper clung to the cliff face"
- Reference tree processes without the word: "Annual rings in the cross-section revealed decades of drought cycles"
- Describe tree structures abstractly: "The canopy intercepted ninety percent of rainfall before it reached the ground"
- Reference tree-specific phenomena: "Mast years flood the ground with acorns", "Samaras spiraled down on the autumn wind"
- Arboriculture without the word "tree": "The certified arborist assessed the veteran oak for structural defects"

---

### Batch 11: Botanical Vocabulary Overlap (Hard Negatives) + Tree Products With Tree Context (Hard Positives)

**25 Negative examples (label=0, subcategory: "botanical_overlap")**

Generate examples that use vocabulary heavily associated with trees but in non-tree contexts:

- **"Trunk"**: elephant trunk, car trunk, steamer trunk, trunk line, trunk show
- **"Ring"**: boxing ring, diamond ring, phone ring, Saturn's rings, ring theory
- **"Crown"**: royal crown, dental crown, crown molding (architectural), Crown (TV show)
- **"Canopy"**: bed canopy, parachute canopy, market stall canopy, aircraft canopy
- **"Grove"**: Grove (as place name or surname), Orange Grove (citrus, not tree focus)
- **"Limb"**: body limb, prosthetic limb, "out on a limb" (idiom), limb darkening (astronomy)
- **"Cone"**: traffic cone, ice cream cone, cone (geometry), cone cell (eye), volcanic cone
- **"Needle"**: sewing needle, compass needle, hypodermic needle, knitting needle
- **"Shade"**: lamp shade, eye shadow/shade, shade (slang for insult), window shade
- **"Timber"**: Timber (exclamation), "shiver me timbers" (pirate phrase)

Also include texts with heavy botanical vocabulary in non-tree plant contexts:
- "Photosynthesis", "chlorophyll", "stomata", "xylem", "phloem" — but about grasses, algae, or other non-tree plants
- "Pruning" a rose bush, "grafting" grape vines, "rooting" cuttings of herbs

**25 Positive examples (label=1, subcategory: "tree_product_explicit")**

Generate examples about tree-derived products where the tree origin IS the focus:

- Tapping maple trees for syrup (the tapping process, not the pancake)
- Cork oak harvesting cycles
- Rubber tree plantations and latex collection
- Paper mills sourcing from managed timber forests
- Charcoal production from specific hardwoods
- Cinnamon bark stripping from Cinnamomum trees
- Naval stores: harvesting pine resin for turpentine and pitch

The distinction from round 1 "material" examples: these should explicitly reference the living tree or the harvesting process, not just the end product.

---

### Batch 12: Stress Test Mix + Edge Cases

**25 Negative examples (label=0, subcategory: "stress_test_neg")**

A diverse mix of the hardest negatives — examples designed to maximally confuse a probe that learned "nature" instead of "trees":

- Descriptions of a garden with flowers, herbs, and vegetables — but no trees
- A beekeeper tending hives in a meadow (bees pollinate trees, but this text is about bees/meadow)
- Mushroom foraging in a field (not a forest)
- Composting kitchen scraps (plant-derived but not tree-focused)
- Descriptions of agricultural fields (crops, irrigation, soil)
- Nature documentary narration about non-tree subjects (birds in grasslands, desert lizards)
- Environmental science about non-forest carbon sinks (peatlands, oceans, soil)
- Landscape architecture focusing on ground cover, stone, and water features (no trees)
- Botanical illustration of non-tree plants
- Herbalism using non-tree plants (chamomile, lavender, echinacea, mint)

**25 Positive examples (label=1, subcategory: "stress_test_pos")**

A diverse mix of tree content in unusual framings:

- Trees in climate change discourse (but framed as carbon policy, not ecology)
- Trees in archaeology (preserved wooden artifacts, dendrochronology dating ruins)
- Trees in music (the wood of a Stradivarius, tonewood selection)
- Trees in children's literature (not just "The Giving Tree" — varied examples)
- Trees in real estate (mature trees adding property value, heritage tree ordinances)
- Trees in disaster contexts (falling trees in storms, wildfire fuel loads)
- Trees in food systems (fruit orchards, nut harvesting, agroforestry)
- Trees in urban planning (street trees, urban heat island mitigation)
- Trees in indigenous knowledge (traditional ecological knowledge about specific species)
- Trees in art (Ansel Adams photography, landscape painting traditions)

---

## Output Format

Same YAML format as round 1:

```yaml
# Batch N: [description]
# Round 2 — targeted at probe generalization gaps

examples:
  - text: "The passage text here."
    label: 1
    subcategory: "unusual_framing"
    notes: "Legal framing of deforestation"

  - text: "Another passage."
    label: 0
    subcategory: "plant_not_tree"
    notes: "Wheat agriculture — no trees"
```

Save as `generated_batch_09.yaml` through `generated_batch_12.yaml`.

---

## Quality Checklist

Before submitting each batch, verify:
- [ ] 25 positive and 25 negative examples (perfect balance)
- [ ] No repeated sentence structures within or across batches
- [ ] Length varies (1-4 sentences, 50-500 characters)
- [ ] Register varies (technical, casual, literary, instructional)
- [ ] **Negative examples contain ZERO references to trees, wood, forests, timber, or tree species**
- [ ] **Negative examples DO use nature/plant vocabulary that overlaps with tree contexts**
- [ ] **Positive examples are genuinely about trees, not just tangentially related**
- [ ] Subcategories correctly assigned
- [ ] No overlap with existing batches 01-08 (check for duplicated species, scenarios, or framings)

---

## What NOT to Generate

Do not generate more of what we already have plenty of:
- Basic "the oak tree stood tall" style positives (covered in batches 01-06)
- Urban/abstract/sports/cooking negatives (covered in batches 01-06)
- leaf/branch/root/bark homonyms (covered in batches 07-08)

Every example in this round should target the specific gaps identified in the adversarial evaluation.
