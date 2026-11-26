"""Regenerate the corrected publications from saved statistics, arrays and existing display layouts."""
import json
from shutil import copyfile

import numpy as np
from threadpoolctl import threadpool_limits

from mathgeom.config import ROOT, load_config, load_yaml
from mathgeom.report import figdata, paper, render, site, style
from mathgeom.report.figures import axes, families, lattice, overview


def main():
    cfg = load_config(ROOT / "configs/mvp.yaml")
    all_results = paper._results("all")
    conf = cfg.models[0]["key"]
    folder = ROOT / "reports/figures"
    style.use_style()
    with threadpool_limits(limits=2):
        site.write_index(cfg, reuse_layout=True)
        reference = json.loads((site.SITE / "index.json").read_text())["fields"]
        saved = json.loads((site.SITE / (conf + ".json")).read_text())
        layer = saved["primary_layer"]
        p = {"arxiv": np.array(reference["arxiv_layout3"])}
        stress = {"arxiv": reference["arxiv_stress"][1]}
        for side in ("trained", "random_init"):
            p[side] = np.array(saved["fields"][side]["layout3"][layer-1])
            stress[side] = saved["fields"][side]["stress"][layer-1][1]
        overview.build(all_results, cfg, folder, saved_field_map=(p, stress))
        axes.build(cfg, all_results["axes"], all_results["mathlib"], {conf: layer}, folder)
        spec = load_yaml("families.yaml")["lattices"]["group_like"]
        names, axioms = list(spec["items"]), spec["axioms"]
        bits = np.array(list(spec["items"].values()))
        for config in (*paper.CROSS, "mvp_variants"):
            current = load_config(ROOT / f"configs/{config}.yaml")
            shapes = json.loads((current.results_dir / "shapes.json").read_text())
            la = json.loads((current.results_dir / "lattice.json").read_text())
            out = folder / current.raw.get("figure_subdir", "")
            for model in (k for k in la if not k.endswith("__random")):
                layer = la[model]["primary_layer"]
                choices = {"trained": model, "random_init": model+"__random"}
                x = {s: figdata.family_acts(current, k, "group_like", names)[:, layer] for s,k in choices.items()}
                off = {s: la[k]["group_like"]["offsets"] for s,k in choices.items()}
                style.save(lattice.draw_explainer(x, names, bits, axioms, off, "commutative", style.pretty(model)), out, "axioms_explainer_"+model)
                style.save(families.draw_angles(current, model, shapes, layer), out, "angles_"+model)
            render.phase_report(current, saved_figures=True)
        paper.build_summary()
        paper.build_audit()
        paper.build_reviews()
        paper.build_readme_block()
        paper.build_paper()
        copyfile(ROOT / "paper/paper.pdf", ROOT / "reports/paper.pdf")


if __name__ == "__main__":
    main()
