"""Scrape Wikipedia for probe training data."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote


def get_wikipedia_api_url(params: dict) -> str:
    """
    Construct Wikipedia API URL with given parameters.

    Args:
        params: Dict of API parameters.

    Returns:
        Full API URL string.
    """
    raise NotImplementedError("TODO")


def search_wikipedia(query: str, num_results: int = 10) -> List[str]:
    """
    Search Wikipedia and return article titles.

    Args:
        query: Search query string.
        num_results: Number of results to return.

    Returns:
        List of article titles.
    """
    raise NotImplementedError("TODO")


def get_article_content(title: str) -> Optional[str]:
    """
    Fetch plain text content of a Wikipedia article.

    Uses the Wikipedia API extracts endpoint to get clean text
    without markup.

    Args:
        title: Article title.

    Returns:
        Article text content, or None if fetch failed.
    """
    raise NotImplementedError("TODO")


def extract_paragraphs(
    text: str,
    min_length: int = 50,
    max_length: int = 500
) -> List[str]:
    """
    Extract paragraphs of suitable length from article text.

    Args:
        text: Full article text.
        min_length: Minimum paragraph character length.
        max_length: Maximum paragraph character length.

    Returns:
        List of paragraph strings.
    """
    raise NotImplementedError("TODO")


def scrape_positive_examples(
    queries: List[str],
    num_paragraphs_per_query: int = 20
) -> List[Dict]:
    """
    Scrape Wikipedia for tree-related content (positive examples).

    Args:
        queries: List of search queries for tree-related topics.
        num_paragraphs_per_query: Target paragraphs per query.

    Returns:
        List of dicts with 'text', 'label', 'source', 'subcategory'.
    """
    raise NotImplementedError("TODO")


def scrape_negative_examples(
    queries: List[str],
    num_paragraphs_per_query: int = 20,
    exclude_keywords: Optional[List[str]] = None
) -> List[Dict]:
    """
    Scrape Wikipedia for non-tree content (negative examples).

    Args:
        queries: List of search queries for non-tree topics.
        num_paragraphs_per_query: Target paragraphs per query.
        exclude_keywords: Words that disqualify a paragraph (tree, forest, etc.)

    Returns:
        List of dicts with 'text', 'label', 'source', 'subcategory'.
    """
    raise NotImplementedError("TODO")


# Default search queries for positive examples (tree-related)
POSITIVE_QUERIES = [
    # Direct tree topics
    "tree species",
    "oak tree",
    "pine tree",
    "maple tree",
    "redwood sequoia",
    "deforestation",
    "reforestation",
    "arboriculture",
    "dendrology",

    # Ecological context
    "forest ecosystem",
    "rainforest",
    "boreal forest",
    "tree canopy",
    "old growth forest",

    # Wood/material
    "lumber industry",
    "woodworking",
    "timber construction",
    "plywood manufacturing",
    "wood pulp paper",

    # Cultural
    "sacred trees mythology",
    "tree symbolism",
    "christmas tree history",
    "bonsai cultivation",
]

# Default search queries for negative examples (non-tree topics)
NEGATIVE_QUERIES = [
    # Nature (non-forest)
    "ocean marine biology",
    "desert ecosystem",
    "coral reef",
    "arctic tundra",
    "volcanic geology",

    # Urban/technology
    "urban planning city",
    "computer architecture",
    "semiconductor manufacturing",
    "automobile engineering",
    "skyscraper construction",

    # Abstract/academic
    "number theory mathematics",
    "quantum mechanics physics",
    "philosophical epistemology",
    "economic theory",
    "statistical analysis",

    # Human activities
    "olympic swimming",
    "culinary techniques cooking",
    "cardiovascular medicine",
    "jazz music history",
    "film cinematography",
]

# Keywords that indicate tree-related content (for filtering negatives)
TREE_KEYWORDS = [
    "tree", "trees", "forest", "forests", "wood", "wooden", "timber",
    "lumber", "oak", "pine", "maple", "birch", "spruce", "fir",
    "redwood", "sequoia", "elm", "ash", "willow", "cedar", "cypress",
    "deforestation", "reforestation", "arborist", "forestry",
    "woodland", "grove", "orchard", "canopy", "trunk", "bark",
    "sapling", "seedling", "logging", "sawmill",
]


def main():
    """Main entry point for Wikipedia scraping."""
    parser = argparse.ArgumentParser(
        description="Scrape Wikipedia for probe training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/concepts/trees/wikipedia_examples.yaml",
        help="Output file path"
    )
    parser.add_argument(
        "--num-positive",
        type=int,
        default=500,
        help="Target number of positive examples"
    )
    parser.add_argument(
        "--num-negative",
        type=int,
        default=500,
        help="Target number of negative examples"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Seconds between API requests"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )

    args = parser.parse_args()

    raise NotImplementedError("TODO: Implement scraping pipeline")


if __name__ == "__main__":
    main()
