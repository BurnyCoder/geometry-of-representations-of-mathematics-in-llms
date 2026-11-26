"""Views of the independent calculations used in the review and the hypothesis ledger."""
import json

from mathgeom.config import ROOT


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|"*len(head)] +
                     ["| " + " | ".join(str(cell).replace("|", r"\|") for cell in row) + " |" for row in rows])


def interval(values):
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


def decisions(s):
    """Report each decision with its units, reference and rule. Use archived scores for fitted probes."""
    def archived(stem):
        return json.loads((ROOT / "results/mvp" / (stem+".json")).read_text())["qwen3.5-2b"]
    r = s["checkpoints"]["qwen3.5-2b"]
    fam = r["trained"]["families"]
    mp, la, sh, hi, st, ax = map(archived, ("map_fields", "lattice", "shapes", "hierarchy", "structures", "axes"))
    out = []
    def add(test, sample, rule, observed, verdict):
        out.append(dict(hypothesis=test, sample=sample, rule=rule, observed=observed, verdict=verdict))
    field = r["trained"]["field"]
    primary = f'{s["sample"]["n"]} concepts, {len(s["sample"]["fields"])} fields'
    add("H-A1, primary", primary, "Whole-field permutation p < .05 and positive paired trained-minus-twin CI",
        f'ρ {field["rho"]:.5f}, twin {r["random_init"]["field"]["rho"]:.5f}, Δ {r["delta"]:.5f}, CI {interval(r["delta_ci"])}',
        "Supported for the chosen arXiv proxy and fixed fields.")
    for key in ("clustering_math", "clustering_control"):
        m = mp[key]
        add("H-A1c, "+key.split("_")[1], f'{m["n"]} concepts in {m["n_fields"]} fields', "LOO centroid accuracy, paired CI above zero versus twin",
            f'{m["trained"]["acc"]:.3f} vs {m["random_init"]["acc"]:.3f}, Δ CI {interval(m["delta_vs_random_init"]["ci"])}',
            "Clustering detected. Granularity differs across domains. Mathematical specificity unresolved.")
    add("H-A2, branch hierarchy", primary, "UPGMA branch match beats shuffled labels and twin",
        f'ρ {field["supertree_rho"]:.3f}, historical shuffle p {hi["coph_match_p_shuffled_tree"]:.4f}, historical twin Δ CI {interval(hi["coph_match_delta_vs_random_ci"])}',
        "Twin interval crosses zero. The conjunction fails on the MVP.")
    add("H-A3, math checkpoint", "Two Qwen2.5 checkpoints, same glossary", "Positive paired Δρ CI",
        f'Δ {s["math_training"]["delta"]:.4f}, CI {interval(s["math_training"]["ci"])}', "Predicted increase unresolved. Equivalence untested.")
    applied = r["applied_labels"]
    add("H-A4, pure/applied", primary, "Secondary association with leading PCs. Post-hoc permute whole fields, preserving count applied",
        f'PC2 |ρ| {applied["absolute_rhos"][1]:.3f}, max-five p {applied["max_five_pc_p"][1]:.4f}', "Association survives this field-level reference.")
    add("H-A4, abstraction", primary, "Secondary rank association after regressing raw scores on frequency and token count",
        f'PC1 ρ {ax["pcs"][0]["rho_abstraction"]:.3f}, saved five-fold readout ρ {ax["cv_ridge_rho_abstraction"]:.3f}', "Association with model ratings. Human judgments unavailable.")
    ml = s["mathlib"]["reviewed"]
    add("H-A4, external proxy", f'{ml["n_matched"]}/{ml["n_math_concepts"]} math concepts', "Post-hoc semantic filtering of rating vs module-depth comparison",
        f'ρ {ml["rho_mean_rating_depth"]:.3f}, CI {interval(ml["ci95"])}', "Positive on the retained subset. Cross-field coverage is sparse.")
    for f in ("days","months","angles"):
        v = fam[f]
        rule = "Closure < 2 and circle ρ > line ρ"
        if f != "angles":
            rule += " and Robinson p < .05"
        add("H-B, "+f, f'{sh[f]["n"]} fixed names', rule, f'closure {v["closure"]:.2f}, circle {v["circle_rho"]:.3f}, line {v["line_rho"]:.3f}',
            "Full-space positive control failed." if f != "angles" else "Specified prediction failed alongside both calendar controls. Latent circularity unresolved.")
    for f in ("ring_chain","space_chain"):
        m, v = s["multiplicity"][f], fam[f]
        add("H-B, "+f, f'{sh[f]["n"]} fixed names', "Both Robinson and LOO order p < .05, Holm, trained Robinson > twin",
            f'Robinson exact p {v["robinson_exact"]["p"]:.5f}, archived LOO p {sh[f]["loo_order_p"]:.5f}, Holm {m["published"]:.4f}/{m["reserved"]:.4f}',
            "Multiplicity-sensitive. Head-word control unresolved." if f == "ring_chain" else "Survives both adjustments for this sequence.")
    for f in ("ngons","dimensions"):
        v=sh[f]["beyond_reference"]
        add("H-B, "+f, f'{sh[f]["n"]} ordered names and number words', "RSA with number reference. Exploratory residual template fit",
            f'ρ {v["rho_fam_ref"]:.3f}, residual p {v["p"]:.3f}', "Positive correspondence. Partial RDM p is sensitive to proxy confounding.")
    add("H-B, number systems", f'{sh["number_systems"]["n"]} names', "Exploratory order permutation", f'Robinson p {sh["number_systems"]["robinson_p"]:.4f}', "Descriptive order on these names.")
    neg=sh["negative_control"]["robinson_p"]
    add("H-B, random lists", f'{neg["n_lists"]} lists of 11', "Rejection fraction above .10 flags a calibration problem",
        f'Robinson rejection fraction {neg["reject_rate_0.05"]:.2f}', "No alarm for the examined nulls. Limited calibration scope.")
    for a,v in fam["offsets"].items():
        m = s["multiplicity"][a]
        add("H-B2, "+a, f'{v["edges"]} one-bit offsets', "Mean pairwise cosine exceeds random-pair null and twin, Holm",
            f'cos {v["cosine"]:.3f}, twin {r["random_init"]["families"]["offsets"][a]["cosine"]:.3f}, Holm {m["published"]:.4f}/{m["reserved"]:.4f}, matched-edge p {v["matched_p"]:.5f}',
            "Survives both adjustments. Lexical alternatives remain." if max(m["published"],m["reserved"]) <= .05 else "Does not pass the published multiplicity correction.")
    g=la["analogy_control"]
    add("H-B2, gender control", "Fixed word pairs", "Consistency exceeds random-pair 95th percentile",
        f'{g["consistency"]:.3f} vs {g["null95"]:.3f}', "Offset method detects the positive control.")
    g = la["group_like"]
    add("H-B2, axiom RDM and analogy", "13 profiles", "Secondary Hamming-RDM coefficient and three-term vector completion",
        f'β {g["axiom_rdm_beta"]:.3f}, p {g["axiom_rdm_p"]:.4f}, top-1 {g["analogy"]["top1"]:.3f} vs chance {g["analogy"]["chance_top1"]:.3f}',
        "Descriptive. Shared names and nuisance misspecification can affect both statistics.")
    sae = archived("sae")
    add("Sparse autoencoder", "MVP at the recorded SAE block", "Exploratory field RSA and centroid accuracy in sparse codes",
        f'ρ {sae["math"]["rsa_sae"]:.3f}, LOO accuracy {sae["math"]["loo_acc_sae"]:.3f}', "Archived aggregate scores. Feature selectivity is in-sample.")
    stg=st["group_objects"]["parallelograms"]
    add("H-B2, related structures", "Self-distributive names, group-object analogies, ring-like transfers", "Exploratory offset comparisons and parent ranks",
        f'group objects cos {stg["cos"]:.3f}, commutativity transfer p {st["transfer"]["commutative"]["p"]:.3f}, identity p {st["transfer"]["identity"]["p"]:.3f}',
        "Group-object offsets also occur in name controls. Ring-like transfer unresolved. Small families have little resolution.")
    cc=s["intervention"]
    add("Chain intervention", f'{cc["space_chain"]["baseline"]["presentations"]} ordered presentations per chain', "Baseline ≥ .70 gates intervention, compare with five random 2-D bases",
        f'ring baseline {cc["ring_chain"]["baseline"]["correct"]}, space baseline {cc["space_chain"]["baseline"]["correct"]}, space correct at s=4 {cc["space_chain"]["chain_s4.0"]["correct"]}',
        "Space passes gate. Separate candidate noise and dependent presentations limit causal interpretation.")
    return out


def render_science(name, s):
    if name == "science_decisions":
        return table(["Hypothesis","Sample","Control or rule","Observation","Decision"], [row.values() for row in s["hypotheses"]])
    if name == "science_fields":
        return table(["Checkpoint","Role","Trained ρ","Twin ρ","Δ, 95% CI","Mantel p","Branch match, twin Δ CI¹"],
            [[k,r["role"],f'{r["trained"]["field"]["rho"]:.5f}',f'{r["random_init"]["field"]["rho"]:.5f}',
              f'{r["delta"]:.5f} {interval(r["delta_ci"])}',f'{r["trained"]["field"]["mantel_p"]:.5f}',interval(r["hierarchy_delta_ci"])] for k,r in s["checkpoints"].items()])
    if name == "science_depth_shift":
        offsets = {r["checked_depth"]-r["original_depth"] for r in s["mathlib"]["dispositions"]}
        if len(offsets) == 1:
            return f"Every checked depth shifted by {next(iter(offsets)):+d}. Their ranks are unchanged, so the change in Spearman ρ comes from the semantic exclusions."
        return f"Checked depth corrections range from {min(offsets):+d} to {max(offsets):+d}."
    if name == "science_matches":
        return table(["Concept at glossary revision","Field","Lean source, original → checked depth","Decision","Reason"],
            [[f'[{r["concept"]}]({r["glossary"]})',r["field"],f'[{r["original_depth"]} → {r["checked_depth"]}]({r["declaration"]})',r["decision"],r["reason"]] for r in s["mathlib"]["dispositions"]])
    raise KeyError(name)
