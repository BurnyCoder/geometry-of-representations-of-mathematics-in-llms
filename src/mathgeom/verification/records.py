"""Reconcile recorded candidate log likelihoods and judge outputs without constructing a model."""
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def intervention(log, families, saved):
    populations = {f: families[f]["items"] for f in ("ring_chain", "space_chain")}
    scores = defaultdict(dict)
    for line in log.open(encoding="utf-8-sig"):
        r = json.loads(line)
        if "causal_condition" not in r:
            continue
        a, b = [r[k].strip() for k in ("continuation_a", "continuation_b")]
        family = next(k for k, members in populations.items() if a in members and b in members)
        items = populations[family]
        right = (r["logp_a"] > r["logp_b"]) == (items.index(a) < items.index(b))
        assert right == r["correct"] and a != b
        group = scores[family, r["causal_condition"]]
        assert (a, b) not in group
        group[a, b] = right
    out = {}
    for family, members in populations.items():
        ref = saved[family]
        expected = {"baseline": ref["baseline_acc"]}
        for scale, row in ref["scales"].items():
            expected[f"chain_s{scale}"] = row["acc_chain_subspace"]
            for j, p in enumerate(row["acc_random_subspace"]):
                expected[f"random{j}_s{scale}"] = p
            np.testing.assert_allclose(np.mean(row["acc_random_subspace"]), row["acc_random_subspace_mean"])
        assert set(expected) == {c for f, c in scores if f == family}
        out[family] = {}
        for condition, mean in expected.items():
            group = scores[family, condition]
            assert len(group) == len(members) * (len(members)-1) == ref["n_questions"]
            count = int(sum(group.values()))
            np.testing.assert_allclose(count/len(group), mean)
            out[family][condition] = {"correct": count, "presentations": len(group)}
    return out


def judges(log, ratings):
    """Digit probabilities were logged rounded; expectations were also logged at full precision."""
    records = defaultdict(dict)
    for line in log.open(encoding="utf-8-sig"):
        r = json.loads(line)
        if "digit_probs" not in r:
            continue
        p = np.asarray(r["digit_probs"])
        np.testing.assert_allclose(p @ np.arange(1, 8), r["expected_rating"], atol=.0015)
        assert r["concept"] not in records[r["judge"]]
        records[r["judge"]][r["concept"]] = (r["expected_rating"], r["p_mass_on_digits"])
    judges = [c for c in ratings if "/" in c]
    assert set(records) == set(judges)
    out = {}
    for j in judges:
        pairs = pd.DataFrame.from_dict(records[j], orient="index", columns=["rating", "mass"]).loc[ratings.concept]
        np.testing.assert_allclose(pairs.rating, ratings[j], atol=1e-10)
        out[j] = {"n": len(pairs), "median_digit_mass": float(pairs.mass.median())}
    return {"by_judge": out, "rank_agreement": spearmanr(ratings[judges]).statistic.tolist()}

def multiplicity(shapes, lattice, nonpositive, draws):
    """Apply Holm's ordered step-down adjustment to the frozen eight slots (Holm 1979)."""
    original = {"H-A1": (1 + nonpositive) / (1 + draws), "angles": 1.0}
    original.update({f: max(shapes[f]["robinson_p"], shapes[f]["loo_order_p"]) for f in ("ring_chain", "space_chain")})
    original.update({a: row["p"] for a, row in lattice["group_like"]["offsets"].items()})
    def adjusted(raw):
        ranked = sorted(raw, key=raw.get)
        values = np.maximum.accumulate([(len(raw)-i)*raw[k] for i,k in enumerate(ranked)])
        return dict(zip(ranked, np.minimum(values, 1.0)))
    first, second = adjusted(original), adjusted(dict(original, **{"H-A1": 1.0}))
    return {k: {"input": p, "published": first[k], "reserved": second[k]} for k,p in original.items()}


def saved_analyses():
    """Check internally testable identities in recorded secondary outputs; retain their evidentiary level."""
    from mathgeom.config import ROOT
    def saved(name):
        return json.loads((ROOT / "results/mvp" / (name+".json")).read_text())["qwen3.5-2b"]
    sae, baseline, shape, ax = map(saved, ("sae", "baselines", "shapes", "axes"))
    for domain in ("math", "control"):
        only = sae["features_"+domain+"_only"]
        assert only + sae["features_shared"] == sae[domain]["n_features_used"]
        assert 0 <= sae[domain]["loo_acc_sae"] <= 1
    for family in ("ngons", "dimensions", "angles", "number_systems", "ring_chain", "space_chain"):
        row = shape[family]
        for stat in ("robinson", "loo_order", "line_rho", "circle_rho"):
            assert -1 <= row[stat] <= 1 and 0 < row[stat+"_p"] <= 1
    for subset in ("all_primary", "fasttext_covered"):
        for comparison in baseline[subset]["boot"].values():
            if isinstance(comparison, dict) and "ci" in comparison:
                assert comparison["ci"][0] <= comparison["ci"][1]
    return {"embedding": {d: {k: baseline[d][k] for k in ("n", "rsa")} for d in ("all_primary", "fasttext_covered")},
            "judge_readout": {"stored_five_fold_ridge_rho": ax["cv_ridge_rho_abstraction"]},
            "sae": {k: sae[k] for k in ("sae_repo", "sae_block", "k", "features_math_only", "features_control_only", "features_shared")}}

def cached_embeddings(artifacts, frame, primary, target):
    """RSA on the saved Qwen embedding matrix and the archived published fastText vocabulary."""
    from mathgeom.analyses.baselines import fasttext_vectors
    from mathgeom.verification.fields import normranks, rdm
    qwen = np.load(artifacts / "baselines_qwen3emb.npy")
    ft, present = fasttext_vectors(artifacts / "baselines/wiki-news-300d-1M.vec.zip", frame.concept.tolist())
    original = saved_analyses()["embedding"]
    out = {}
    for group, mask in (("all_primary", primary), ("fasttext_covered", primary & present)):
        labels = frame.loc[mask, "field"].to_numpy()
        keep = [i for i,f in enumerate(target["names"]) if f in labels]
        fields = [target["names"][i] for i in keep]
        iu = np.triu_indices(len(fields), 1)
        ref = normranks(np.array(target["distance"])[np.ix_(keep, keep)][iu])
        assert int(mask.sum()) == original[group]["n"]
        out[group] = {}
        for name, vec in (("qwen3_embedding", qwen), ("fasttext", ft)):
            if name not in original[group]["rsa"]:
                continue
            x = vec[mask].astype(float)
            items = [x[labels == f] for f in fields]
            distances = rdm(np.stack([v.mean(0) for v in items]), [len(v) for v in items])[iu]
            rho = float(normranks(distances) @ ref)
            np.testing.assert_allclose(rho, original[group]["rsa"][name], atol=1e-10)
            out[group][name] = rho
    return out

EXTRACTION_LOGS = ("20261029-062417_mvp", "20261029-071103_robustness", "20261029-085551_gemma")


def extraction_order(artifacts, frame):
    """Match every field prompt to the ordered CSV row, template and final-template token count."""
    from mathgeom.config import ROOT
    from mathgeom.prompts.templates import FIELD_NEUTRAL
    out = {}
    concepts = list(frame.concept)
    domains = ["mathematics" if k == "math" else "science" for k in frame.kind]
    for prefix in EXTRACTION_LOGS:
        path = ROOT / "logs" / (prefix+"_extract_prompts.jsonl")
        with path.open(encoding="utf-8-sig") as source:
            for line in source:
                record = json.loads(line)
                tag = record.get("tag", "")
                if not tag.endswith("/field_neutral"):
                    continue
                key = tag.split("/")[0]
                if key not in out:
                    with np.load(artifacts / "acts" / key / "field_neutral.npz") as arrays:
                        out[key] = {"n": 0, "tokens": arrays["n_tokens"], "first": record["time"]}
                row = out[key]
                i, t = divmod(row["n"], len(FIELD_NEUTRAL))
                assert (record["concept"], record["template"], record["prompt"]) == (
                    concepts[i], t, FIELD_NEUTRAL[t].format(D=domains[i], X=concepts[i]))
                if t == len(FIELD_NEUTRAL)-1:
                    assert record["concept_tokens"] == row["tokens"][i]
                row["n"] += 1
        for row in out.values():
            assert row["n"] == len(concepts) * len(FIELD_NEUTRAL)
    for row in out.values():
        del row["tokens"]
    assert len(out) == 10
    return out
