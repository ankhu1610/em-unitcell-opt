"""Launch-only checks: no eigensolves or mutation of old experiments."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))


def invoke(arguments, evidence):
    import run_pwe_multistart_v1 as cli
    return (
        mock.patch.object(sys, "argv", ["run_pwe_multistart_v1.py", *arguments]),
        mock.patch.object(cli, "reviewed_evidence", return_value=evidence),
        mock.patch.object(cli, "MultistartExperiment"),
    )


def test_diagnostic_never_launches(capsys):
    import run_pwe_multistart_v1 as cli
    argv, guard, constructor = invoke([], {"passed": True, "launch_blockers": []})
    with argv, guard, constructor as experiment:
        cli.main()
        experiment.assert_not_called()
    assert "launch_blockers" in capsys.readouterr().out


@pytest.mark.parametrize("arguments,evidence,error", [
    (["--run"], {"passed": True, "launch_blockers": []}, SystemExit),
    (["--run", "--acknowledge-exploratory-review"],
     {"passed": False, "launch_blockers": ["stale"]}, ValueError),
    (["--run", "--acknowledge-exploratory-review"],
     {"passed": True, "launch_blockers": ["inconsistent readiness"]}, ValueError),
    (["--run", "--acknowledge-exploratory-review"],
     {"passed": True, "launch_blockers": []}, SystemExit),
    (["--run", "--acknowledge-exploratory-review", "--until-utc", "2020-01-01T00:00:00Z"],
     {"passed": True, "launch_blockers": []}, SystemExit),
    (["--run", "--acknowledge-exploratory-review", "--until-utc", "2090-01-01T00:00:00"],
     {"passed": True, "launch_blockers": []}, SystemExit),
])
def test_failed_gate_or_invalid_deadline_cannot_construct(arguments, evidence, error):
    import run_pwe_multistart_v1 as cli
    argv, guard, constructor = invoke(arguments, evidence)
    with argv, guard, constructor as experiment, pytest.raises(error):
        cli.main()
    experiment.assert_not_called()


def test_excessive_runtime_cannot_construct():
    import run_pwe_multistart_v1 as cli
    date = datetime.now(timezone.utc)+timedelta(hours=5)
    args = ["--run", "--acknowledge-exploratory-review", "--until-utc", date.isoformat()]
    argv, guard, constructor = invoke(args, {"passed": True, "launch_blockers": []})
    with argv, guard, constructor as experiment, pytest.raises(SystemExit):
        cli.main()
    experiment.assert_not_called()
