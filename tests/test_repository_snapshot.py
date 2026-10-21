"""Standard-library tests for documentation generation, not scientific validation."""
import hashlib
import importlib.util
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repository_snapshot", ROOT / "scripts/build_repository_snapshot.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SnapshotTests(unittest.TestCase):
    def test_hashes_cover_source_bytes(self):
        records, hashes = MODULE.load_sources(ROOT)
        self.assertEqual(set(records), set(MODULE.SOURCES))
        for path, digest in hashes.items():
            self.assertEqual(digest, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())

    def test_snapshot_is_current(self):
        rendered = MODULE.render(*MODULE.load_sources(ROOT))
        self.assertEqual((ROOT / "docs/RESULTS_SNAPSHOT.md").read_text(encoding="utf-8"), rendered)

    def test_render_reads_outcome_not_hardcoded_success(self):
        records, hashes = MODULE.load_sources(ROOT)
        records["atlas"]["successful_cases"] = 0
        records["atlas"]["cases"][0]["success"] = False
        rendered = MODULE.render(records, hashes)
        self.assertIn(f"Successful cases: 0/{len(records['atlas']['cases'])}", rendered)
        self.assertIn("| False |", rendered)

    def test_new_documentation_links_resolve(self):
        paths = ["README.md", "docs/FORWARD_AND_INVERSE_DESIGN.md",
                 "docs/REPRODUCIBILITY.md", "docs/REPOSITORY_PREPARATION.md",
                 "docs/RESULTS_SNAPSHOT.md"]
        for relative in paths:
            source = ROOT / relative
            for target in re.findall(r"\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
                if "://" in target or target.startswith("#"):
                    continue
                local = target.split("#", 1)[0]
                self.assertTrue((source.parent / local).exists(), f"Broken link in {relative}: {target}")


if __name__ == "__main__":
    unittest.main()
