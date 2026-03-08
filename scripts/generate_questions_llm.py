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

from src.utils.concept_config import (
    load_concept_config, get_label_names, get_field_names,
    get_pair_id_ranges, get_concept_name,
)

# Prefix pool for elicitation style assignment
ELICITATION_STYLES = [
    "default", "concise", "verbose", "eli5",
    "numbered_list", "no_keyword", "academic", "casual",
]

# === Default (trees) constants for backward compat ===
_DEFAULT_GROUP_B_TOPICS = [
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

_DEFAULT_GROUP_C_SUBCATEGORIES = [
    "shrubs/bushes", "ferns", "mosses/bryophytes", "grasses/crops",
    "fungi/mushrooms", "bamboo", "cacti/succulents", "seaweed/marine algae",
    "flowers/herbaceous plants", "lichens",
]

_DEFAULT_GROUP_D_SUBCATEGORIES = {
    "CS data structures (binary trees, B-trees, red-black)": 20,
    "decision trees and ML": 15,
    "family trees and genealogy": 15,
    "phylogenetic trees and cladistics": 15,
    "parse trees, syntax trees, directory trees": 10,
    "wood-adjacent non-tree topics (log cabins, furniture, boats, paper)": 25,
    "metaphorical vocabulary (branching rivers, root causes, bark of dog)": 15,
    "game/probability trees, skill trees": 10,
}

_DEFAULT_GROUP_E_SUBCATEGORIES = {
    "dendrochronology": 16,
    "lumber/timber industry": 16,
    "bark/sap chemistry": 16,
    "forest ecology": 16,
    "arboriculture": 15,
    "wood engineering": 16,
    "cultural/historical": 15,
    "fruit/nut production": 15,
}


def get_group_topics(cc):
    """Get group B-E topics/subcategories from concept config or defaults."""
    if cc is None:
        return (_DEFAULT_GROUP_B_TOPICS, _DEFAULT_GROUP_C_SUBCATEGORIES,
                _DEFAULT_GROUP_D_SUBCATEGORIES, _DEFAULT_GROUP_E_SUBCATEGORIES)
    return (
        cc.get("group_b_topics", _DEFAULT_GROUP_B_TOPICS),
        cc.get("group_c_subcategories", _DEFAULT_GROUP_C_SUBCATEGORIES),
        cc.get("group_d_subcategories", _DEFAULT_GROUP_D_SUBCATEGORIES),
        cc.get("group_e_subcategories", _DEFAULT_GROUP_E_SUBCATEGORIES),
    )


def load_prompt_template(group: str, data_dir: Path = None) -> str:
    """Load the generation prompt template for a group.

    Looks for concept-specific template in data_dir first, then falls back
    to generic templates in data/generation_prompt_templates/.
    """
    # Try concept-specific template
    if data_dir:
        concept_path = data_dir / "generation_prompts" / f"group_{group.lower()}.txt"
        if concept_path.exists():
            with open(concept_path, "r") as f:
                return f.read()

    # Try generic template
    generic_path = PROJECT_ROOT / "data" / "generation_prompt_templates" / f"group_{group.lower()}.txt"
    if generic_path.exists():
        with open(generic_path, "r") as f:
            return f.read()

    # Fallback to trees-specific template
    trees_path = PROJECT_ROOT / "data" / "concepts" / "trees_qa_v2" / "generation_prompts" / f"group_{group.lower()}.txt"
    with open(trees_path, "r") as f:
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


def validate_yaml_output(text: str, expected_count: int, start_id: int, cc=None) -> dict:
    """Parse and validate the YAML output from Claude."""
    # Strip any markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
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

    pos_field, neg_field = get_field_names(cc)
    # Accept either concept-specific or generic field names
    for i, pair in enumerate(pairs):
        if "pair_id" not in pair:
            raise ValueError(f"Pair {i} missing field: pair_id")
        has_pos = pos_field in pair or "positive_question" in pair
        has_neg = neg_field in pair or "negative_question" in pair
        if not has_pos:
            raise ValueError(f"Pair {i} missing positive question field ({pos_field} or positive_question)")
        if not has_neg:
            raise ValueError(f"Pair {i} missing negative question field ({neg_field} or negative_question)")
        if "question_type" not in pair:
            raise ValueError(f"Pair {i} missing field: question_type")
        if pair["question_type"] not in ("factual", "explanatory", "comparative", "process"):
            raise ValueError(f"Pair {i} invalid question_type: {pair['question_type']}")

    return data


def generate_group_b(batch_num: int, data_dir: Path, tracker_path: str, cc=None):
    """Generate one batch of Group B pairs (25 pairs for one concept topic)."""
    group_b_topics, _, _, _ = get_group_topics(cc)
    pair_id_ranges = get_pair_id_ranges(cc)

    if batch_num < 1 or batch_num > len(group_b_topics):
        raise ValueError(f"Group B batch must be 1-{len(group_b_topics)}, got {batch_num}")

    topic = group_b_topics[batch_num - 1]
    count = 25
    start_id = pair_id_ranges["B"][0] + (batch_num - 1) * 25

    print(f"Group B batch {batch_num}: topic='{topic}', {count} pairs, IDs {start_id}-{start_id + count - 1}")

    # Load used domains
    used_domains = load_used_domains(tracker_path)

    # Generate unique non-tree domains
    domains = generate_non_tree_domains(count, used_domains, seed=42 + batch_num)

    # Load and fill template
    template = load_prompt_template("B", data_dir)
    prompt = template.format(
        tree_topic=topic,
        concept_topic=topic,
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
    data = validate_yaml_output(response, count, start_id, cc)

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


def generate_group_c(subcategory: str, data_dir: Path, tracker_path: str, cc=None):
    """Generate Group C pairs for one subcategory (15 pairs)."""
    _, group_c_subcats, _, _ = get_group_topics(cc)
    pair_id_ranges = get_pair_id_ranges(cc)

    if subcategory not in group_c_subcats:
        raise ValueError(f"Unknown Group C subcategory: {subcategory}. Must be one of: {group_c_subcats}")

    subcat_idx = group_c_subcats.index(subcategory)
    count = 15
    start_id = pair_id_ranges["C"][0] + subcat_idx * 15

    print(f"Group C subcategory='{subcategory}', {count} pairs, IDs {start_id}-{start_id + count - 1}")

    template = load_prompt_template("C", data_dir)
    prompt = template.format(
        subcategory=subcategory,
        count=count,
        start_id=start_id,
    )

    print(f"  Calling Claude API...")
    response = call_claude_api(prompt)

    print(f"  Validating output...")
    data = validate_yaml_output(response, count, start_id, cc)

    styles = assign_elicitation_styles(count, seed=start_id)
    for pair, style in zip(data["pairs"], styles):
        pair["elicitation_style"] = style
        pair["base_pair_ref"] = None

    safe_name = subcategory.replace("/", "_").replace(" ", "_")
    batch_file = data_dir / f"contrastive_batch_C_{safe_name}.yaml"
    with open(batch_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved to {batch_file}")


def generate_group_d(subcategory: str, data_dir: Path, tracker_path: str, cc=None):
    """Generate Group D pairs for one subcategory."""
    _, _, group_d_subcats, _ = get_group_topics(cc)
    pair_id_ranges = get_pair_id_ranges(cc)

    matching = [(k, v) for k, v in group_d_subcats.items() if subcategory.lower() in k.lower()]
    if not matching:
        raise ValueError(f"Unknown Group D subcategory: {subcategory}. Available: {list(group_d_subcats.keys())}")

    subcat_name, count = matching[0]

    # Compute start_id by summing counts of preceding subcategories
    cumulative = 0
    for k, v in group_d_subcats.items():
        if k == subcat_name:
            break
        cumulative += v
    start_id = pair_id_ranges["D"][0] + cumulative

    print(f"Group D subcategory='{subcat_name}', {count} pairs, IDs {start_id}-{start_id + count - 1}")

    template = load_prompt_template("D", data_dir)
    prompt = template.format(
        subcategory=subcat_name,
        count=count,
        start_id=start_id,
    )

    print(f"  Calling Claude API...")
    response = call_claude_api(prompt)

    print(f"  Validating output...")
    data = validate_yaml_output(response, count, start_id, cc)

    styles = assign_elicitation_styles(count, seed=start_id)
    for pair, style in zip(data["pairs"], styles):
        pair["elicitation_style"] = style
        pair["base_pair_ref"] = None

    safe_name = subcat_name.split("(")[0].strip().replace(" ", "_").replace("/", "_")
    batch_file = data_dir / f"contrastive_batch_D_{safe_name}.yaml"
    with open(batch_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved to {batch_file}")


def generate_group_e(subcategory: str, data_dir: Path, tracker_path: str, cc=None):
    """Generate Group E pairs for one subcategory."""
    _, _, _, group_e_subcats = get_group_topics(cc)
    pair_id_ranges = get_pair_id_ranges(cc)

    matching = [(k, v) for k, v in group_e_subcats.items() if subcategory.lower() in k.lower()]
    if not matching:
        raise ValueError(f"Unknown Group E subcategory: {subcategory}. Available: {list(group_e_subcats.keys())}")

    subcat_name, count = matching[0]

    cumulative = 0
    for k, v in group_e_subcats.items():
        if k == subcat_name:
            break
        cumulative += v
    start_id = pair_id_ranges["E"][0] + cumulative

    print(f"Group E subcategory='{subcat_name}', {count} pairs, IDs {start_id}-{start_id + count - 1}")

    # Load used domains for negative side
    used_domains = load_used_domains(tracker_path)
    domains = generate_non_tree_domains(count, used_domains, seed=start_id)

    template = load_prompt_template("E", data_dir)
    prompt = template.format(
        subcategory=subcat_name,
        count=count,
        start_id=start_id,
    )

    print(f"  Calling Claude API...")
    response = call_claude_api(prompt)

    print(f"  Validating output...")
    data = validate_yaml_output(response, count, start_id, cc)

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
                        help="Output directory (default: data/concepts/{concept}_qa_v2)")
    parser.add_argument("--concept-config", type=str, default=None,
                        help="Path to concept config YAML (for non-trees concepts)")
    args = parser.parse_args()

    # Load concept config if provided
    cc = None
    if args.concept_config:
        cc_path = Path(args.concept_config)
        if not cc_path.is_absolute():
            cc_path = PROJECT_ROOT / cc_path
        cc = load_concept_config(str(cc_path))

    # Determine data dir
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif cc:
        data_dir = PROJECT_ROOT / "data" / "concepts" / f"{cc['concept_name']}_qa_v2"
    else:
        data_dir = PROJECT_ROOT / "data" / "concepts" / "trees_qa_v2"

    data_dir.mkdir(parents=True, exist_ok=True)
    tracker_path = str(data_dir / "used_domains.json")

    # Get topics from concept config
    group_b_topics, group_c_subcats, group_d_subcats, group_e_subcats = get_group_topics(cc)

    if args.group == "B":
        if args.all_batches:
            for batch_num in range(1, len(group_b_topics) + 1):
                print(f"\n{'='*60}")
                generate_group_b(batch_num, data_dir, tracker_path, cc)
        elif args.batch:
            generate_group_b(args.batch, data_dir, tracker_path, cc)
        else:
            parser.error("Group B requires --batch N or --all-batches")

    elif args.group == "C":
        if args.all_batches:
            for subcat in group_c_subcats:
                print(f"\n{'='*60}")
                generate_group_c(subcat, data_dir, tracker_path, cc)
        elif args.subcategory:
            generate_group_c(args.subcategory, data_dir, tracker_path, cc)
        else:
            parser.error("Group C requires --subcategory NAME or --all-batches")

    elif args.group == "D":
        if args.all_batches:
            for subcat in group_d_subcats:
                print(f"\n{'='*60}")
                generate_group_d(subcat, data_dir, tracker_path, cc)
        elif args.subcategory:
            generate_group_d(args.subcategory, data_dir, tracker_path, cc)
        else:
            parser.error("Group D requires --subcategory NAME or --all-batches")

    elif args.group == "E":
        if args.all_batches:
            for subcat in group_e_subcats:
                print(f"\n{'='*60}")
                generate_group_e(subcat, data_dir, tracker_path, cc)
        elif args.subcategory:
            generate_group_e(args.subcategory, data_dir, tracker_path, cc)
        else:
            parser.error("Group E requires --subcategory NAME or --all-batches")

    print("\nDone!")


if __name__ == "__main__":
    main()
