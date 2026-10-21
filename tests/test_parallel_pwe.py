import sys
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import run_pwe_parallel as runner


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_read_retries_permission_error(self):
        operation = Mock(side_effect=[PermissionError("busy"), {"ok": True}])
        with patch.object(runner.time, "sleep"):
            self.assertEqual(runner.checkpoint_io(operation), {"ok": True})
        self.assertEqual(operation.call_count, 2)

    def test_optional_monitoring_defers_persistent_access_error(self):
        operation = Mock(side_effect=PermissionError("busy"))
        with patch.object(runner.time, "sleep"):
            self.assertIsNone(runner.checkpoint_io(operation, optional=True))
            with self.assertRaises(PermissionError):
                runner.checkpoint_io(operation)

    def test_monitoring_does_not_hide_other_failures(self):
        with self.assertRaises(ValueError):
            runner.checkpoint_io(Mock(side_effect=ValueError("bad data")), optional=True)

    def test_complete_file_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/"predictions").mkdir()
            path = out/"predictions/D001.json"
            path.write_text(json.dumps({"fingerprint": "fixed", "frequencies_GHz": [[1., 2.]]}))
            before = path.read_bytes()
            with patch.object(runner, "OUT", out):
                runner.worker({"design_id": "D001"}, {"k": [.1]}, "fixed")
            self.assertEqual(before, path.read_bytes())

    def test_resume_keeps_saved_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/"predictions").mkdir()
            path = out/"predictions/D001.json"
            prefix = [[1.25, 2.5]]
            path.write_text(json.dumps({"fingerprint": "fixed", "frequencies_GHz": prefix,
                                        "diagnostics": [{"old": True}], "elapsed_compute_seconds": 1.}))
            with patch.object(runner, "OUT", out), \
                 patch.object(runner, "tensor_for", return_value=(None, None)), \
                 patch.object(runner, "lowest_two", return_value=(np.array([3., 4.]), {"new": True})):
                runner.worker({"design_id": "D001"}, {"k": [.1, .2], "a_um": 242.5}, "fixed")
            after = json.loads(path.read_text())
            self.assertEqual(after["frequencies_GHz"][:1], prefix)
            self.assertEqual(after["diagnostics"][0], {"old": True})
            self.assertEqual(len(after["frequencies_GHz"]), 2)


if __name__ == "__main__":
    unittest.main()
