"""Ground-truth field distances from arXiv cross-listing statistics.

Global context: MSC2020 is flat above its 2-digit codes, so "tree distance" between top-level
fields would be just same/different. Instead we measure how often mathematicians themselves
cross-list papers between fields: for fields A, B (each a set of arXiv categories), with
S_A = set of papers listing any category of A, similarity = Ochiai/cosine
|S_A ∩ S_B| / sqrt(|S_A| |S_B|), distance = 1 − similarity. Data: the CC0 mirror
`librarian-bots/arxiv-metadata-snapshot` of the Kaggle arXiv metadata
(https://www.kaggle.com/datasets/Cornell-University/arxiv), read column-wise from parquet so
only the `categories` column is downloaded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
REPO = "librarian-bots/arxiv-metadata-snapshot"


def load_categories(cache: Path) -> tuple[list[str], str]:
    """Return the per-paper category strings and the dataset commit sha (cached as JSON)."""
    if cache.exists():
        js = json.loads(cache.read_text())
        return js["categories"], js["revision"]
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    sha = HfApi().dataset_info(REPO).sha  # pin the exact snapshot used
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{REPO}@{sha}/data/train-*.parquet"))
    cats: list[str] = []
    for f in files:  # column projection → range requests for just this column
        cats += pq.read_table(f, columns=["categories"], filesystem=fs).column("categories").to_pylist()
        log.info("read %s (%d papers so far)", f.split("/")[-1], len(cats))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"revision": sha, "categories": cats}))
    return cats, sha


def field_distance_matrix(categories: list[str], fields: dict[str, list[str]]) -> tuple[np.ndarray, dict]:
    """Ochiai distance between fields from paper-level category co-occurrence.

    Returns (k×k distance matrix in `fields` order, stats dict with paper counts/overlaps).
    """
    names = list(fields)
    member = np.zeros((len(categories), len(names)), dtype=bool)
    for p, c in enumerate(categories):
        cs = set(c.split())
        for k, n in enumerate(names):
            member[p, k] = bool(cs.intersection(fields[n]))
    M = member.astype(np.int64)
    co = M.T @ M  # co[a,b] = papers in both fields; diagonal = field sizes
    size = np.diag(co).astype(float)
    sim = co / np.sqrt(np.outer(size, size))
    D = 1 - sim
    np.fill_diagonal(D, 0.0)
    return D, {"names": names, "papers_per_field": size.astype(int).tolist(), "co": co.tolist()}
