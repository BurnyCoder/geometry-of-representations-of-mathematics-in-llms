"""Mathlib depth matcher on a synthetic three-module library: import depth, namespaces, name matching."""

from mathgeom.data.mathlib_depth import declarations, import_depth, match_concepts, module_imports

FILES = {
    "Mathlib/A.lean": "/-\nCopyright\n-/\nmodule\n\n/-! docs: import Mathlib.C must not count -/\nstructure Foo where\n  x : Nat\n",
    "Mathlib/B.lean": "module\n\npublic import Mathlib.A\n\nnamespace X\n\nclass Bar where\n\nsection S\ndef Baz := 1\nend S\n\nend X\n\nclass Bar where\n",
    "Mathlib/Sub/C.lean": "module\n\npublic import Mathlib.A\nimport Mathlib.B\n\n@[simp] noncomputable def CompactSpace := 1\ntheorem _root_.lemmaQ : True := trivial\n",
}


def _lib(tmp_path):
    for rel, text in FILES.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "overview.yaml").write_text("Algebra:\n  Sub:\n    foo thing: 'Foo'\n    missing: '~'\n", encoding="utf-8")
    return tmp_path


def test_import_depth_and_namespaces(tmp_path):
    root = _lib(tmp_path)
    imps = module_imports(root)
    assert imps == {"Mathlib.A": [], "Mathlib.B": ["Mathlib.A"], "Mathlib.Sub.C": ["Mathlib.A", "Mathlib.B"]}
    assert import_depth(imps) == {"Mathlib.A": 0, "Mathlib.B": 1, "Mathlib.Sub.C": 2}
    full = {d[0] for d in declarations(root)}
    assert {"Foo", "X.Bar", "X.Baz", "Bar", "CompactSpace", "lemmaQ"} <= full


def test_match_yaml_camelcase_and_ambiguity(tmp_path):
    root = _lib(tmp_path)
    m = match_concepts(["foo thing", "compact space", "bar", "bar thing", "missing"], [root / "docs" / "overview.yaml"], declarations(root))
    assert m["foo thing"] == {"decl": "Foo", "module": "Mathlib.A", "source": "mathlib_yaml"}
    assert m["compact space"]["module"] == "Mathlib.Sub.C" and m["compact space"]["source"] == "camelcase"
    assert "bar" not in m  # single words are never name-matched
    assert "missing" not in m  # '~' = not formalized
