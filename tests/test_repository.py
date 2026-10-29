"""Dependency-free archive checks, not model or performance validation."""

import ast
import hashlib
import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "experiments" / "historical"


class RepositoryChecks(unittest.TestCase):
    def test_python_syntax(self):
        for path in ROOT.rglob("*.py"):
            if ".git" not in path.parts:
                with self.subTest(path=path.relative_to(ROOT)):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_historical_snapshot_integrity(self):
        manifest = json.loads((HISTORICAL / "source_checksums.json").read_text())
        self.assertEqual(set(manifest), {p.name for p in HISTORICAL.glob("*.py")})
        self.assertEqual(len(manifest), 6)
        for name, expected in manifest.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((HISTORICAL / name).read_bytes()).hexdigest(), expected)

    def test_generated_submission_templates_parse(self):
        tree = ast.parse((HISTORICAL / "package_rank1.py").read_text())
        templates = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"MODEL_TEMPLATE", "SOLUTION_TEMPLATE"}:
                        templates[target.id] = ast.literal_eval(node.value)
        self.assertEqual(set(templates), {"MODEL_TEMPLATE", "SOLUTION_TEMPLATE"})
        model = templates["MODEL_TEMPLATE"].format(
            backbone="convnext_zepto_rms.ra4_e3600_r224_in1k",
            embedding_dim=256,
            class_name="TestEmbeddingModel",
        )
        ast.parse(model)
        for use_tta in ("True", "False"):
            ast.parse(templates["SOLUTION_TEMPLATE"].replace("__USE_TTA__", use_tta))

    def test_relative_markdown_links_exist(self):
        for path in ROOT.rglob("*.md"):
            for target in re.findall(r"\[[^\]]+\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                with self.subTest(document=path.relative_to(ROOT), target=target):
                    resolved = (path.parent / unquote(parsed.path)).resolve()
                    self.assertTrue(resolved.is_relative_to(ROOT), target)
                    self.assertTrue(resolved.exists(), target)

    def test_reported_results_are_consistent_with_readme(self):
        results = json.loads((ROOT / "results" / "official_final_results.json").read_text())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(set(results["tracks"]), {"T1", "T2"})
        for track in results["tracks"].values():
            self.assertLessEqual(track["rank"], track["ranked_teams"])
            for key in ("VER@0.1%", "VER@1%", "EER", "AUC"):
                self.assertGreaterEqual(track[key], 0)
                self.assertLessEqual(track[key], 1)
                self.assertIn(f"{track[key]:.4f}", readme)


if __name__ == "__main__":
    unittest.main()
