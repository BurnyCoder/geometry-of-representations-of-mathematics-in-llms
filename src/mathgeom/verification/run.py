"""Run with uv run --no-sync python -m mathgeom.verification.run [--write]. CPU statistics only."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from mathgeom.config import ROOT, load_yaml, load_config
from mathgeom.verification import fields, geometry, mathlib, records


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def progress(message):
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def midpoint(path, mask=None):
    """Read one existing array, release the other layers, and return no model object."""
    with np.load(path) as arrays:
        last = arrays["last"]
        layer = round((last.shape[1]-1)/2)
        return last[slice(None) if mask is None else mask, layer].astype(float), arrays["n_tokens"], layer


def collect():
    cfg = load_config(ROOT / "configs/mvp.yaml")
    data = ROOT / "data"
    artifacts = cfg.artifacts
    concepts = pd.read_csv(data / "concepts.csv")
    assert not concepts.concept.str.casefold().duplicated().any()
    assert (concepts.primary == ~(concepts.leak_own | concepts.eponym | concepts.polysemous)).all()
    # Each field consumes two integer seeds before the original Bernoulli pilot assignment.
    split_rng = np.random.default_rng(cfg.seed)
    split_rng.integers(1 << 31, size=2 * concepts.field.nunique())
    np.testing.assert_array_equal(concepts.split.eq("pilot"), split_rng.random(len(concepts)) < cfg.pilot_fraction)
    assert not concepts.multi_field.any()
    mask = (concepts.kind.eq("math") & concepts.primary & concepts.split.eq("confirmatory")).to_numpy()
    sample = concepts[mask]
    spec = load_yaml("fields.yaml")["math"]
    fam = load_yaml("families.yaml")
    target = read(data / "field_distance.json")
    progress("Recounting the arXiv category co-listings")
    arxiv = fields.check_target(artifacts / "arxiv_categories.json", spec, target)
    design = fields.reference_design(sample.field.to_numpy(), target["names"], np.asarray(target["distance"]),
                                     {f["id"]: f["branch"] for f in spec})
    index = read(artifacts / "acts/family_index.json")
    for family, spec_row in fam["chains"].items():
        assert [name for group,name,_ in index if group == family] == spec_row["items"]
    out = {"source_revision": "43dcda70a7f6093e946fbc975c88cd65a7d12e5f", "prereg_tag": "prereg-v1",
           "arxiv": arxiv, "sample": {"n": len(sample), "fields": sample.field.value_counts().to_dict(),
           "all_splits": concepts.groupby(["kind","split"]).size().unstack().to_dict()}, "checkpoints": {}}
    draws = {}
    for config in ("mvp", "robustness", "gemma"):
        folder = ROOT / "results" / config
        maps, shapes, lattices, hierarchy = [read(folder / (k+".json")) for k in ("map_fields","shapes","lattice","hierarchy")]
        for key, mp in maps.items():
            progress("Checking cached field labels, bootstrap, orderings and edges for " + key)
            r = {"role": "confirmatory" if config == "mvp" else "robustness", "layer": mp["primary_layer"]}
            boot, hierarchy_boot = {}, {}
            for side, suffix in (("trained",""), ("random_init","__random")):
                cache = artifacts / "acts" / (key+suffix)
                x, tokens, layer = midpoint(cache / "field_neutral.npz", mask)
                assert layer == mp["primary_layer"] and len(tokens) == len(concepts)
                metrics, boot[side], hierarchy_boot[side] = fields.checkpoint(x, design)
                np.testing.assert_allclose(metrics["rho"], mp["math_primary"]["rsa"][side], atol=1e-12)
                np.testing.assert_allclose(metrics["supertree_rho"], hierarchy[key]["coph_match"][side], atol=1e-12)
                if side == "trained" and config == "mvp":
                    r["applied_labels"] = fields.applied_labels(x, sample, tokens[mask])
                fx, _, family_layer = midpoint(cache / "family.npz")
                assert layer == family_layer and len(index) == len(fx)
                by_family = geometry.family_check(fx, index, fam["lattices"]["group_like"], shapes[key+suffix], lattices[key+suffix])
                r[side] = {"field": metrics, "families": by_family}
            diff = boot["trained"] - boot["random_init"]
            r["hierarchy_delta_ci"] = np.percentile(hierarchy_boot["trained"] - hierarchy_boot["random_init"], [2.5, 97.5]).tolist()
            r["delta"] = r["trained"]["field"]["rho"] - r["random_init"]["field"]["rho"]
            r["delta_ci"] = np.percentile(diff, [2.5,97.5]).tolist()
            np.testing.assert_allclose(r["delta_ci"], mp["math_primary"]["rsa_boot"]["delta_vs_random_init"]["ci"], atol=1e-12)
            r["positive_bootstrap_draws"] = int(np.sum(diff > 0))
            out["checkpoints"][key] = r
            draws[key] = boot["trained"]
    a, b = "qwen2.5-math-1.5b", "qwen2.5-1.5b"
    out["math_training"] = {"delta": out["checkpoints"][a]["trained"]["field"]["rho"] - out["checkpoints"][b]["trained"]["field"]["rho"],
                            "ci": np.percentile(draws[a]-draws[b], [2.5,97.5]).tolist()}
    np.testing.assert_allclose(out["math_training"]["ci"], read(ROOT/"results/robustness/compare.json")[a+"_vs_"+b]["boot"]["delta_vs_"+b]["ci"])
    key = "qwen3.5-2b"
    out["multiplicity"] = records.multiplicity(read(ROOT/"results/mvp/shapes.json")[key],
        read(ROOT/"results/mvp/lattice.json")[key], len(draws[key]) - out["checkpoints"][key]["positive_bootstrap_draws"], len(draws[key]))
    out["saved_analyses"] = records.saved_analyses()
    out["embedding_checks"] = records.cached_embeddings(artifacts, concepts, mask, target)
    out["minimal_upper_bounds"] = geometry.minimal_upper_bounds(fam["lattices"]["group_like"]["items"], "commutative magma","unital magma")
    progress("Checking the Mathlib meanings, judge distributions and intervention candidate scores")
    out["mathlib"] = mathlib.assess(artifacts / "mathlib4", read(ROOT / "results/mvp/mathlib.json"))
    out["judges"] = records.judges(ROOT / "logs/20261029-094116_mvp_judges_prompts.jsonl", pd.read_csv(data/"judge_ratings.csv"))
    out["intervention"] = records.intervention(ROOT / "logs/20261029-124502_mvp_causal_prompts.jsonl", fam["chains"],
                                                read(ROOT/"results/mvp/causal.json")["qwen3.5-2b"])
    from mathgeom.verification.tables import decisions
    out["hypotheses"] = decisions(out)
    return out


def equal_evidence(new, old):
    """Compare every key and record, allowing only ordinary floating-point roundoff."""
    if isinstance(new, dict):
        assert new.keys() == old.keys()
        for k in new:
            equal_evidence(new[k], old[k])
    elif isinstance(new, list):
        assert len(new) == len(old)
        for a,b in zip(new,old):
            equal_evidence(a,b)
    elif isinstance(new, float):
        np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-11)
    else:
        assert new == old, (new,old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh only the derived verification JSON")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        results = collect()
    from mathgeom.verification.coverage import inventory
    evidence = ROOT / "results/scientific_audit"
    path = evidence / "verification.json"
    files = inventory()
    if args.write:
        path.parent.mkdir(exist_ok=True)
        (evidence / "file_coverage.json").write_text(json.dumps(files, indent=2)+"\n", encoding="utf-8")
        path.write_text(json.dumps(results, indent=2, allow_nan=False, ensure_ascii=False)+"\n", encoding="utf-8")
    else:
        equal_evidence(results, read(path))
        equal_evidence(files, read(evidence / "file_coverage.json"))
        from mathgeom.verification.publications import check
        check(results)
    progress("Verified the independent scientific calculations")


if __name__ == "__main__":
    main()
