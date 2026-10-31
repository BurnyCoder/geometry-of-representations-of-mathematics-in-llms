"""Single entry point: `uv run python run.py <phase|all> --config configs/mvp.yaml`.

Phases: the keys of PHASES below, in order (calibration, concepts, arxiv, extract, map, …, lexical, twins),
then `report` (per config) and `paper` (cross-config). `all` runs every phase of PHASES plus report; the
README lists which phases each config actually ran. Each phase reads earlier phases' outputs, so any
phase can be re-run on its own.
"""

import argparse

from mathgeom import manifest, pipeline
from mathgeom.config import ROOT, load_config
from mathgeom.logio import setup_logging

PHASES = {
    "calibration": pipeline.phase_calibration,  # simulation study of RDM-test type-I error → results/<cfg>/calibration.json
    "concepts": pipeline.phase_concepts,  # Wikipedia glossaries (pinned revisions) → data/concepts.csv
    "arxiv": pipeline.phase_arxiv,  # arXiv cross-listing → data/field_distance.json
    "extract": pipeline.phase_extract,  # residual-stream activations → artifacts/acts/<model>/
    "map": pipeline.phase_map,  # H-A1 field map → results/<cfg>/map_fields.json
    "baselines": pipeline.phase_baselines,  # fastText / Qwen3-Embedding field maps vs LLM; infini-gram covariate → baselines.json
    "smds": pipeline.phase_smds,  # secondary H-B: SMDS line/circle fits vs label-shuffled refits → smds.json
    "hierarchy": pipeline.phase_hierarchy,  # H-A2 cophenetic correlation, Gromov δ, Gaussian null → hierarchy.json
    "shapes": pipeline.phase_shapes,  # H-B chains/circles → results/<cfg>/shapes.json
    "chainfreq": pipeline.phase_chainfreq,  # plan H-B frequency control: chain order beyond Zipf and token count → chain_frequency.json
    "lattice": pipeline.phase_lattice,  # H-B2 axiom lattices → results/<cfg>/lattice.json
    "structures": pipeline.phase_structures,  # H-B2 tests 4–5: axiom transfer, Lie/topological/algebraic groups → structures.json
    "judges": pipeline.phase_judges,  # H-A4 covariate: LLM-judge abstraction ratings → data/judge_ratings.csv
    "mathlib": pipeline.phase_mathlib,  # H-A4 judge validation: rating vs Mathlib import depth, coverage per field → mathlib.json
    "axes": pipeline.phase_axes,  # H-A4: PCs vs pure/applied & abstraction → results/<cfg>/axes.json
    "causal": pipeline.phase_causal,  # H-B causal: chain-subspace vs random-subspace noise → results/<cfg>/causal.json
    "sae": pipeline.phase_sae,  # exploratory Qwen-Scope SAE check → results/<cfg>/sae.json
    "posthoc": pipeline.phase_posthoc,  # POST-HOC: ABTT depth sweep + per-PC field variance → results/<cfg>/posthoc.json
    "compare": pipeline.phase_compare,  # H-A3 paired model contrasts (config `comparisons`) → results/<cfg>/compare.json
    "lexical": pipeline.phase_lexical,  # independent audit: word-form baselines, within-block order, nonce modifiers, BCa → lexical.json
    "twins": pipeline.phase_twins,  # independent audit: twin statistics across random-init seeds (configs/twin_seeds.yaml) → twins.json
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=[*PHASES, "report", "paper", "all"])
    ap.add_argument("--config", default="configs/mvp.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(ROOT / "logs", f"{cfg.name}_{args.phase}")
    if args.phase in ("report", "all"):
        from mathgeom.report.render import phase_report

        PHASES["report"] = phase_report
    if args.phase == "paper":  # needs results of all configs (mvp, robustness, gemma)
        from mathgeom.report.paper import build_audit, build_cross_figures, build_paper, build_readme_block, build_summary

        build_cross_figures()
        build_readme_block()
        build_summary()
        build_audit()
        build_paper()
        return
    todo = [*PHASES] if args.phase == "all" else [args.phase]
    for p in todo:
        PHASES[p](cfg)
        manifest.record(cfg, p, args.config)  # git SHA, config hash, seed, library versions


if __name__ == "__main__":
    main()
