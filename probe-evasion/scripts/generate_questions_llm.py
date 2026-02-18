"""
Generate contrastive question pairs using Claude API.

Generates question pair texts for groups B, C, D, E in the v2 dataset.
Each group has specific prompt templates and constraints. Runs locally (no GPU).

Usage:
    python scripts/generate_questions_llm.py --group B --batch 1
    python scripts/generate_questions_llm.py --group C --subcategory "shrubs/bushes" --batch 1
    python scripts/generate_questions_llm.py --group B --all-batches
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prefix pool for elicitation style assignment
ELICITATION_STYLES = [
    "default", "concise", "verbose", "eli5",
    "numbered_list", "no_keyword", "academic", "casual",
]

# Group B: 15 tree topics x 25 pairs each = 375 pairs
GROUP_B_TREE_TOPICS = [
    "tree biology (photosynthesis, reproduction, genetics)",
    "tree anatomy (bark, roots, leaves, rings, crown)",
    "tree ecology (forests, succession, symbiosis, mycorrhizae)",
    "tree species diversity (oak, pine, maple, baobab, sequoia, birch)",
    "dendrochronology and age dating",
    "silviculture and forestry management",
    "urban trees and arboriculture",
    "tree products (lumber, paper, resin, latex, fruit, syrup, charcoal)",
    "trees and climate (carbon sequestration, deforestation)",
    "cultural and historical trees (sacred groves, Yggdrasil, Bodhi tree)",
    "tree diseases and pests (Dutch elm, emerald ash borer)",
    "wood properties and engineering (hardness, grain, moisture)",
    "tree adaptations (desert, mangrove, alpine, tropical)",
    "trees in art, literature, and symbolism",
    "tree conservation and endangered species",
]

# Group C: 10 subcategories x 15 pairs each = 150 pairs
GROUP_C_SUBCATEGORIES = [
    "shrubs/bushes", "ferns", "mosses/bryophytes", "grasses/crops",
    "fungi/mushrooms", "bamboo", "cacti/succulents", "seaweed/marine algae",
    "flowers/herbaceous plants", "lichens",
]

# Group D: 8 subcategories with varying counts = 125 pairs total
GROUP_D_SUBCATEGORIES = {
    "CS data structures (binary trees, B-trees, red-black)": 20,
    "decision trees and ML": 15,
    "family trees and genealogy": 15,
    "phylogenetic trees and cladistics": 15,
    "parse trees, syntax trees, directory trees": 10,
    "wood-adjacent non-tree topics (log cabins, furniture, boats, paper)": 25,
    "metaphorical vocabulary (branching rivers, root causes, bark of dog)": 15,
    "game/probability trees, skill trees": 10,
}

# Group E: 8 subcategories x ~16 pairs each = 125 pairs total
GROUP_E_SUBCATEGORIES = {
    "dendrochronology": 16,
    "lumber/timber industry": 16,
    "bark/sap chemistry": 16,
    "forest ecology": 16,
    "arboriculture": 15,
    "wood engineering": 16,
    "cultural/historical": 15,
    "fruit/nut production": 15,
}

# Pair ID ranges per group
PAIR_ID_RANGES = {
    "A": (0, 24),
    "B": (25, 399),
    "C": (400, 549),
    "D": (550, 674),
    "E": (675, 799),
}


def load_prompt_template(group: str) -> str:
    """Load the generation prompt template for a group."""
    template_path = PROJECT_ROOT / "data" / "concepts" / "trees_qa_v2" / "generation_prompts" / f"group_{group.lower()}.txt"
    with open(template_path, "r") as f:
        return f.read()


def load_used_domains(tracker_path: str) -> list:
    """Load already-used non-tree domains from tracker file."""
    if os.path.exists(tracker_path):
        with open(tracker_path, "r") as f:
            return json.load(f)
    return []


def save_used_domains(tracker_path: str, domains: list):
    """Save used non-tree domains to tracker file."""
    with open(tracker_path, "w") as f:
        json.dump(sorted(set(domains)), f, indent=2)


def assign_elicitation_styles(count: int, seed: int = 42) -> list:
    """Assign roughly balanced elicitation styles to `count` pairs."""
    rng = random.Random(seed)
    styles = []
    # Create a balanced pool
    full_cycles = count // len(ELICITATION_STYLES)
    remainder = count % len(ELICITATION_STYLES)
    pool = ELICITATION_STYLES * full_cycles + rng.sample(ELICITATION_STYLES, remainder)
    rng.shuffle(pool)
    return pool


def generate_non_tree_domains(count: int, used: list, seed: int = 42) -> list:
    """Generate a list of diverse non-tree domains, avoiding already used ones."""
    # Large pool of potential domains across STEM, humanities, arts, sports, trades
    all_domains = [
        "marine biology", "astronomy", "cooking", "history", "metallurgy",
        "dance", "ceramics", "telecommunications", "forensics", "textiles",
        "nuclear physics", "cryptography", "plumbing", "aviation", "beekeeping",
        "sculpture", "automotive engineering", "meteorology", "photography", "economics",
        "glaciology", "opera", "robotics", "quilting", "volcanology",
        "chess strategy", "dentistry", "cartography", "perfumery", "archaeology",
        "jazz music", "veterinary medicine", "glassblowing", "seismology", "origami",
        "electrical engineering", "calligraphy", "parasitology", "watchmaking", "surfing",
        "philosophy", "dermatology", "blacksmithing", "oceanography", "stand-up comedy",
        "civil engineering", "ballet", "microbiology", "gem cutting", "curling",
        "sociology", "ophthalmology", "welding", "paleontology", "magic tricks",
        "acoustics", "knitting", "entomology", "architecture", "fencing sport",
        "neuroscience", "pottery", "ornithology", "graphic design", "wrestling",
        "pharmacology", "bookbinding", "ichthyology", "interior design", "bowling",
        "immunology", "leatherworking", "herpetology", "fashion design", "archery",
        "endocrinology", "basket weaving", "primatology", "typography", "kayaking",
        "cardiology", "candle making", "malacology", "film directing", "rock climbing",
        "pulmonology", "soap making", "arachnology", "animation", "skateboarding",
        "nephrology", "dyeing and printing", "mammalogy", "video game design", "skiing",
        "gastroenterology", "embroidery", "cetology", "sound engineering", "cycling",
        "rheumatology", "stained glass", "limnology", "screenwriting", "rowing",
        "oncology", "mosaic art", "pedology", "choreography", "badminton",
        "hematology", "woodturning", "bryology", "music composition", "table tennis",
        "toxicology", "paper marbling", "mycology", "theater directing", "golf",
        "virology", "silk painting", "ethology", "podcast production", "sailing",
        "epidemiology", "macrame", "nematology", "documentary filmmaking", "swimming",
        "pathology", "felting", "phycology", "journalism", "marathon running",
        "radiology", "batik", "helminthology", "museum curation", "volleyball",
        "anesthesiology", "tatting lace", "protozoology", "urban planning", "handball",
        "orthopedics", "weaving", "palynology", "food science", "water polo",
        "pediatrics", "gilding", "speleology", "linguistics", "cricket sport",
        "geriatrics", "enameling", "lidar technology", "anthropology", "rugby",
        "psychiatry", "marquetry", "hydroponics", "political science", "baseball",
        "urology", "crochet", "aquaponics", "rhetoric", "basketball",
        "neurology", "glass etching", "renewable energy", "semiotics", "hockey",
        "obstetrics", "copper smithing", "3D printing", "ethics", "lacrosse",
        "allergy medicine", "upholstery", "quantum computing", "aesthetics", "polo",
        "sports medicine", "tin smithing", "nanotechnology", "logic", "equestrian",
        "emergency medicine", "fresco painting", "blockchain", "pragmatics", "diving",
        "plastic surgery", "mural painting", "satellite technology", "hermeneutics", "triathlon",
        "pain management", "relief carving", "augmented reality", "phenomenology", "biathlon",
        "sleep medicine", "pyrography", "machine learning hardware", "existentialism", "squash sport",
        "addiction medicine", "cloisonne", "supply chain logistics", "stoicism", "luge",
        "tropical medicine", "sgraffito", "cybersecurity", "epistemology", "bobsled",
        "occupational therapy", "encaustic painting", "telecommunications infrastructure", "metaphysics", "pentathlon",
        "audiology", "damascene metalwork", "aerospace engineering", "semaphore", "skeleton sport",
        "prosthetics", "niello work", "biomedical devices", "morphology linguistics", "curling",
        "kinesiology", "tole painting", "nuclear engineering", "discourse analysis", "freestyle wrestling",
        "chiropractic", "reverse glass painting", "chemical engineering", "corpus linguistics", "synchronized swimming",
        "acupuncture", "scrimshaw", "mining engineering", "translation studies", "water skiing",
        "optometry", "intarsia", "petroleum engineering", "comparative literature", "speed skating",
        "podiatry", "repoussé", "environmental engineering", "creative writing", "figure skating",
        "speech therapy", "pewter casting", "industrial design", "dramaturgy", "ice hockey",
        "physical therapy", "lost wax casting", "materials science", "musicology", "Alpine skiing",
        "massage therapy", "chasing metal", "structural engineering", "art history", "cross-country skiing",
        "nutrition science", "mokume gane", "geotechnical engineering", "comparative religion", "snowboarding",
        "biostatistics", "kintsugi", "fire protection engineering", "cultural studies", "ski jumping",
        "public health", "raku pottery", "transportation engineering", "media studies", "biathlon",
    ]

    used_set = set(d.lower().strip() for d in used)
    available = [d for d in all_domains if d.lower().strip() not in used_set]

    rng = random.Random(seed)
    if len(available) < count:
        print(f"WARNING: Only {len(available)} unique domains available, need {count}. Some may be less diverse.")
        # Pad with generated variants
        while len(available) < count:
            available.append(f"domain_{len(available)}")

    selected = rng.sample(available, count)
    return selected


def call_claude_api(prompt: str, max_tokens: int = 8192) -> str:
    """Call Claude API to generate question pairs."""
    import anthropic

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def validate_yaml_output(text: str, expected_count: int, start_id: int) -> dict:
    """Parse and validate the YAML output from Claude."""
    # Strip any markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    data = yaml.safe_load(text)

    if not isinstance(data, dict) or "pairs" not in data:
        raise ValueError("YAML output missing 'pairs' key")

    pairs = data["pairs"]
    if len(pairs) != expected_count:
        raise ValueError(f"Expected {expected_count} pairs, got {len(pairs)}")

    required_fields = [
        "pair_id", "tree_question", "non_tree_question",
        "question_type", "non_tree_domain", "group",
    ]
    for i, pair in enumerate(pairs):
        for field in required_fields:
            if field not in pair:
                raise ValueError(f"Pair {i} missing field: {field}")
        if pair["question_type"] not in ("factual", "explanatory", "comparative", "process"):
            raise ValueError(f"Pair {i} invalid question_type: {pair['question_type']}")

    return data


def generate_group_b(batch_num: int, data_dir: Path, tracker_path: str):
    """Generate one batch of Group B pairs (25 pairs for one tree topic)."""
    if batch_num < 1 or batch_num > 15:
        raise ValueError(f"Group B batch must be 1-15, got {batch_num}")

    topic = GROUP_B_TREE_TOPICS[batch_num - 1]
    count = 25
    start_id = PAIR_ID_RANGES["B"][0] + (batch_num - 1) * 25

    print(f"Group B batch {batch_num}: topic='{topic}', {count} pairs, IDs {start_id}-{start_id + count - 1}")

    # Load used domains
    used_domains = load_used_domains(tracker_path)

    # Generate unique non-tree domains
    domains = generate_non_tree_domains(count, used_domains, seed=42 + batch_num)

    # Load and fill template
    template = load_prompt_template("B")
    prompt = template.format(
        tree_topic=topic,
        domains="\n".join(f"- {d}" for d in domains),
        used_domains="\n".join(f"- {d}" for d in used_domains) if used_domains else "(none yet)",
        count=count,
        start_id=start_id,
    )

    # Call Claude
    print(f"  Calling Claude API...")
    response = call_claude_api(prompt)

    # Validate
    print(f"  Validating output...")
    data = validate_yaml_output(response, count, start_id)

    # Assign elicitation styles
    styles = assign_elicitation_styles(count, seed=start_id)
    for pair, style in zip(data["pairs"], styles):
        pair["elicitation_style"] = style
        pair["similarity"] = pair.get("similarity", False)
        pair["base_pair_ref"] = None

    # Save
    batch_file = data_dir / f"contrastive_batch_B_{batch_num:02d}.yaml"
    with open(batch_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved to {batch_file}")

    # Update domain tracker
    new_domains = [p["non_tree_domain"] for p in data["pairs"]]
    used_domains.extend(new_domains)
    save_used_domains(tracker_path, used_domains)
    print(f"  Updated domain tracker ({len(set(used_domains))} total domains)")


def generate_group_c(subcategory: str, data_dir: Path, tracker_path: str):
    """Generate Group C pairs for one subcategory (15 pairs)."""
    if subcategory not in GROUP_C_SUBCATEGORIES:
        raise ValueError(f"Unknown Group C subcategory: {subcategory}. Must be one of: {GROUP_C_SUBCATEGORIES}")

    subcat_idx = GROUP_C_SUBCATEGORIES.index(subcategory)
    count = 15
    start_id = PAIR_ID_RANGES["C"][0] + subcat_idx * 15

    print(f"Group C subcategory='{subcategory}', {count} pairs, IDs {start_id}-{start_id + count - 1}")

    template = load_prompt_template("C")
    prompt = template.format(
        subcategory=subcategory,
        count=count,
        start_id=start_id,
    )

    print(f"  Calling Claude API...")
    response = call_claude_api(prompt)

    print(f"  Validating output...")
    data = validate_yaml_output(response, count, start_id)

    styles = assign_elicitation_styles(count, seed=start_id)
    for pair, style in zip(data["pairs"], styles):
        pair["elicitation_style"] = style
        pair["base_pair_ref"] = None

    safe_name = subcategory.replace("/", "_").replace(" ", "_")
    batch_file = data_dir / f"contrastive_batch_C_{safe_name}.yaml"
    with open(batch_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved to {batch_file}")


def generate_group_d(subcategory: str, data_dir: Path, tracker_path: str):
    """Generate Group D pairs for one subcategory."""
    matching = [(k, v) for k, v in GROUP_D_SUBCATEGORIES.items() if subcategory.lower() in k.lower()]
    if not matching:
        raise ValueError(f"Unknown Group D subcategory: {subcategory}. Available: {list(GROUP_D_SUBCATEGORIES.keys())}")

    subcat_name, count = matching[0]

    # Compute start_id by summing counts of preceding subcategories
    cumulative = 0
    for k, v in GROUP_D_SUBCATEGORIES.items():
        if k == subcat_name:
            break
        cumulative += v
    start_id = PAIR_ID_RANGES["D"][0] + cumulative

    print(f"Group D subcategory='{subcat_name}', {count} pairs, IDs {start_id}-{start_id + count - 1}")

    template = load_prompt_template("D")
    prompt = template.format(
        subcategory=subcat_name,
        count=count,
        start_id=start_id,
    )

    print(f"  Calling Claude API...")
    response = call_claude_api(prompt)

    print(f"  Validating output...")
    data = validate_yaml_output(response, count, start_id)

    styles = assign_elicitation_styles(count, seed=start_id)
    for pair, style in zip(data["pairs"], styles):
        pair["elicitation_style"] = style
        pair["base_pair_ref"] = None

    safe_name = subcat_name.split("(")[0].strip().replace(" ", "_").replace("/", "_")
    batch_file = data_dir / f"contrastive_batch_D_{safe_name}.yaml"
    with open(batch_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved to {batch_file}")


def generate_group_e(subcategory: str, data_dir: Path, tracker_path: str):
    """Generate Group E pairs for one subcategory."""
    matching = [(k, v) for k, v in GROUP_E_SUBCATEGORIES.items() if subcategory.lower() in k.lower()]
    if not matching:
        raise ValueError(f"Unknown Group E subcategory: {subcategory}. Available: {list(GROUP_E_SUBCATEGORIES.keys())}")

    subcat_name, count = matching[0]

    cumulative = 0
    for k, v in GROUP_E_SUBCATEGORIES.items():
        if k == subcat_name:
            break
        cumulative += v
    start_id = PAIR_ID_RANGES["E"][0] + cumulative

    print(f"Group E subcategory='{subcat_name}', {count} pairs, IDs {start_id}-{start_id + count - 1}")

    # Load used domains for non-tree side
    used_domains = load_used_domains(tracker_path)
    domains = generate_non_tree_domains(count, used_domains, seed=start_id)

    template = load_prompt_template("E")
    prompt = template.format(
        subcategory=subcat_name,
        count=count,
        start_id=start_id,
    )

    print(f"  Calling Claude API...")
    response = call_claude_api(prompt)

    print(f"  Validating output...")
    data = validate_yaml_output(response, count, start_id)

    styles = assign_elicitation_styles(count, seed=start_id)
    for pair, style in zip(data["pairs"], styles):
        pair["elicitation_style"] = style
        pair["base_pair_ref"] = None

    safe_name = subcat_name.replace("/", "_").replace(" ", "_")
    batch_file = data_dir / f"contrastive_batch_E_{safe_name}.yaml"
    with open(batch_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved to {batch_file}")

    # Update domain tracker
    new_domains = [p["non_tree_domain"] for p in data["pairs"]]
    used_domains.extend(new_domains)
    save_used_domains(tracker_path, used_domains)


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive question pairs using Claude API")
    parser.add_argument("--group", type=str, required=True, choices=["B", "C", "D", "E"],
                        help="Which data group to generate")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch number (Group B only, 1-15)")
    parser.add_argument("--subcategory", type=str, default=None,
                        help="Subcategory name (Groups C, D, E)")
    parser.add_argument("--all-batches", action="store_true",
                        help="Generate all batches/subcategories for the group")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Output directory (default: data/concepts/trees_qa_v2)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "concepts" / "trees_qa_v2"
    data_dir.mkdir(parents=True, exist_ok=True)
    tracker_path = str(data_dir / "used_domains.json")

    if args.group == "B":
        if args.all_batches:
            for batch_num in range(1, 16):
                print(f"\n{'='*60}")
                generate_group_b(batch_num, data_dir, tracker_path)
        elif args.batch:
            generate_group_b(args.batch, data_dir, tracker_path)
        else:
            parser.error("Group B requires --batch N or --all-batches")

    elif args.group == "C":
        if args.all_batches:
            for subcat in GROUP_C_SUBCATEGORIES:
                print(f"\n{'='*60}")
                generate_group_c(subcat, data_dir, tracker_path)
        elif args.subcategory:
            generate_group_c(args.subcategory, data_dir, tracker_path)
        else:
            parser.error("Group C requires --subcategory NAME or --all-batches")

    elif args.group == "D":
        if args.all_batches:
            for subcat in GROUP_D_SUBCATEGORIES:
                print(f"\n{'='*60}")
                generate_group_d(subcat, data_dir, tracker_path)
        elif args.subcategory:
            generate_group_d(args.subcategory, data_dir, tracker_path)
        else:
            parser.error("Group D requires --subcategory NAME or --all-batches")

    elif args.group == "E":
        if args.all_batches:
            for subcat in GROUP_E_SUBCATEGORIES:
                print(f"\n{'='*60}")
                generate_group_e(subcat, data_dir, tracker_path)
        elif args.subcategory:
            generate_group_e(args.subcategory, data_dir, tracker_path)
        else:
            parser.error("Group E requires --subcategory NAME or --all-batches")

    print("\nDone!")


if __name__ == "__main__":
    main()
