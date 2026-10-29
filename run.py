"""Single entry point: `uv run python run.py <phase|all> --config configs/mvp.yaml`.

Phases (in order): concepts → arxiv → extract → map → shapes → lattice → report.
Each phase reads the previous phase's outputs, so any phase can be re-run on its own.
"""

import argparse

from mathgeom import pipeline
from mathgeom.config import ROOT, load_config
from mathgeom.logio import setup_logging

PHASES = {
    "calibration": pipeline.phase_calibration,  # simulation study of RDM-test type-I error → results/<cfg>/calibration.json
    "concepts": pipeline.phase_concepts,  # Wikipedia glossaries (pinned revisions) → data/concepts.csv
    "arxiv": pipeline.phase_arxiv,  # arXiv cross-listing → data/field_distance.json
    "extract": pipeline.phase_extract,  # residual-stream activations → artifacts/acts/<model>/
    "map": pipeline.phase_map,  # H-A1 field map → results/<cfg>/map_fields.json
    "shapes": pipeline.phase_shapes,  # H-B chains/circles → results/<cfg>/shapes.json
    "lattice": pipeline.phase_lattice,  # H-B2 axiom lattices → results/<cfg>/lattice.json
    "judges": pipeline.phase_judges,  # H-A4 covariate: LLM-judge abstraction ratings → data/judge_ratings.csv
    "axes": pipeline.phase_axes,  # H-A4: PCs vs pure/applied & abstraction → results/<cfg>/axes.json
    "causal": pipeline.phase_causal,  # H-B causal: chain-subspace vs random-subspace noise → results/<cfg>/causal.json
    "sae": pipeline.phase_sae,  # exploratory Qwen-Scope SAE check → results/<cfg>/sae.json
    "posthoc": pipeline.phase_posthoc,  # POST-HOC: ABTT depth sweep + per-PC field variance → results/<cfg>/posthoc.json
    "compare": pipeline.phase_compare,  # H-A3 paired model contrasts (config `comparisons`) → results/<cfg>/compare.json
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
        from mathgeom.report.paper import build_paper, build_summary

        build_summary()
        build_paper()
        return
    todo = [*PHASES] if args.phase == "all" else [args.phase]
    for p in todo:
        PHASES[p](cfg)


if __name__ == "__main__":
    main()
