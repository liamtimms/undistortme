"""Real-binaries end-to-end smoke test for FSL TOPUP integration.

This test is intentionally excluded from the default ``pytest`` run (which
filters ``not slow and not needs_fsl`` per pytest.ini).  Run it explicitly:

    PY -m pytest tests/test_smoke_fsl.py -m "slow or needs_fsl" -v

WHAT IT TESTS
-------------
* Build a two-echo, one-volume synthetic BIDS run with 32x32x16 NIfTI files
  (the default conftest tiny_nii 8x8x8 is too small for FSL topup's b02b0.cnf
  subsampling steps; 32x32x16 works reliably with pervol.cnf).
* Wire corr_pipeline.check_dict with topup=True, everything else False.
* Call cp.process_run() with pervol.cnf so topup actually runs.
* Assert corrected output files (*desc-undistorted*) exist and are loadable via
  nibabel with finite data.

No numerical quality assertions are made — this is a wiring / integration
check only.

NOTE ON CONFIG
--------------
``pervol.cnf`` (from the repo root) is used instead of ``b02b0.cnf`` because
b02b0.cnf's subsampling schedule requires larger images.  pervol.cnf runs
topup in ~5-8 seconds on a 32x32x16 phantom.

NOTE ON ENVIRONMENT
-------------------
FSL must be discoverable: either ``topup`` is already on PATH, or ``$FSLDIR``
is set (its ``share/fsl/bin`` / ``bin`` subdirectory is then prepended to
PATH).  Otherwise the test skips.  FSLOUTPUTTYPE=NIFTI is set so FSL emits
.nii not .nii.gz, matching the glob patterns used by run_topup.
"""

import glob
import json
import os
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from undistortme import pipeline as cp

# ---------------------------------------------------------------------------
# Marks — excluded from default run; both marks needed to be safe
# ---------------------------------------------------------------------------
pytestmark = [pytest.mark.slow, pytest.mark.needs_fsl]

# Config that works on small phantoms — relative to repo root
_REPO_ROOT = Path(__file__).parent.parent
PERVOL_CNF = str(_REPO_ROOT / "configs" / "pervol.cnf")


# ---------------------------------------------------------------------------
# Module-level skip if topup is not on PATH (after trying $FSLDIR)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _skip_if_no_fsl(monkeypatch):
    """Ensure topup is reachable (PATH, else $FSLDIR), or skip.

    With UNDISTORTME_REQUIRE_FSL=1 a missing FSL FAILS instead of skipping —
    CI uses this so a broken image can never pass its smoke gate by skipping.
    """
    if shutil.which("topup") is None:
        fsldir = os.environ.get("FSLDIR", "")
        candidates = [
            os.path.join(fsldir, "share", "fsl", "bin"),
            os.path.join(fsldir, "bin"),
        ] if fsldir else []
        for bin_dir in candidates:
            if os.path.isfile(os.path.join(bin_dir, "topup")):
                monkeypatch.setenv(
                    "PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
                break
    if shutil.which("topup") is None:
        msg = "FSL topup not found: put it on PATH or set $FSLDIR"
        if os.environ.get("UNDISTORTME_REQUIRE_FSL"):
            pytest.fail(msg)
        pytest.skip(msg)


# ---------------------------------------------------------------------------
# Local tree builder — 32x32x16 volumes (too big for the default 8x8x8
# tiny_nii fixture, but small enough to be fast with pervol.cnf)
# ---------------------------------------------------------------------------

def _make_large_nii(path: Path, shape=(32, 32, 16), seed: int = 0) -> str:
    """Write a real uncompressed NIfTI at *path* and return path as str."""
    rng = np.random.default_rng(seed)
    data = rng.random(shape, dtype=np.float32)
    aff = np.eye(4)
    aff[0, 0] = 3.0
    aff[1, 1] = 3.0
    aff[2, 2] = 3.0
    nib.save(nib.Nifti1Image(data, aff), str(path))
    return str(path)


def _make_smoke_bids_run(tmp_path: Path):
    """Build a two-echo, one-volume BIDS run tree with 32x32x16 NIfTIs.

    Returns a dict compatible with the conftest bids_run factory output:
    output_dir, deriv_dir, subject, session, run, run_dir, tmp_path.
    """
    subject = "sub-01"
    session = "ses-01"
    run = "run-01"

    out = tmp_path / "out"
    deriv = tmp_path / "deriv"
    run_dir = out / subject / session / run
    run_dir.mkdir(parents=True)

    for e in [1, 2]:
        stem = f"{subject}_{session}_{run}_echo-{e}"
        sidecar = {
            "EchoNumber": e,
            "EchoTime": 0.07 + (e - 1) * 0.02,
            "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.106487,
            "PulseSequenceName": "ep_seg_35",
            "SeriesDescription": "ME_GRE",
            "ImageType": ["ORIGINAL", "PRIMARY", "M"],
            "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
        }
        (run_dir / f"{stem}.json").write_text(json.dumps(sidecar))
        _make_large_nii(run_dir / f"{stem}_1.nii", seed=e * 100 + 1)

    return {
        "output_dir": str(out),
        "deriv_dir": str(deriv),
        "subject": subject,
        "session": session,
        "run": run,
        "run_dir": str(run_dir),
        "tmp_path": str(tmp_path),
    }


# ---------------------------------------------------------------------------
# The smoke test
# ---------------------------------------------------------------------------

def test_topup_smoke(tmp_path, monkeypatch):
    """End-to-end: process_run with real FSL topup on a 32x32x16 phantom.

    Wall time with pervol.cnf: ~5-8 s (dominated by topup itself).
    """
    # --- environment setup ---
    # (PATH already carries FSL_BIN via the autouse _skip_if_no_fsl fixture.)
    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI")

    # --- build synthetic run ---
    run_info = _make_smoke_bids_run(tmp_path)

    # --- wire check_dict (real dispatchers, no recorder) ---
    check_dict = {
        "dcm2niix": False,
        "topup": True,
        "topup_multithread": False,   # keep single-threaded for reproducibility
        "slice": False,
        "match": False,
        "dryrun": False,
        "two_echo": False,
        "mask": False,
    }
    monkeypatch.setattr(cp, "check_dict", check_dict, raising=False)

    # --- run the pipeline ---
    t0 = time.time()
    cp.process_run(
        run_info["subject"],
        run_info["session"],
        run_info["run"],
        run_info["output_dir"],
        run_info["deriv_dir"],
        PERVOL_CNF,
    )
    elapsed = time.time() - t0
    print(f"\n[smoke] process_run elapsed: {elapsed:.1f}s")

    # --- assertions ---
    topup_dir = Path(run_info["deriv_dir"]) / "undistortme" / "whole-volume" / \
        run_info["subject"] / run_info["session"] / run_info["run"]

    corr_files = sorted(glob.glob(str(topup_dir / "*desc-undistorted*")))

    # Helpful failure message listing actual directory contents
    dir_contents = sorted(glob.glob(str(topup_dir / "*"))) if topup_dir.exists() else []
    assert len(corr_files) > 0, (
        f"No *desc-undistorted* files found in {topup_dir}.\n"
        f"Directory contents ({len(dir_contents)} items):\n"
        + "\n".join(f"  {p}" for p in dir_contents)
    )

    for path in corr_files:
        img = nib.load(path)
        data = img.get_fdata()
        assert np.isfinite(data).all(), (
            f"Non-finite values in corrected output: {path}"
        )

    print(f"[smoke] corrected files found: {[os.path.basename(f) for f in corr_files]}")


# ---------------------------------------------------------------------------
# Slice-mode smoke test — additionally requires slicenii + combinenii.
# Caught a real incompatibility: slicenii 0.2.0 release binaries slice fine
# but their combinenii cannot recombine the pipeline's per-slice outputs
# (needs > 0.2.0 with combinenii axis-guessing / sorting fixes).
# ---------------------------------------------------------------------------

def test_topup_slice_smoke(tmp_path, monkeypatch):
    """End-to-end slice mode: slicenii -> per-slice topup -> combinenii."""
    if shutil.which("slicenii") is None or shutil.which("combinenii") is None:
        msg = "slicenii/combinenii not found on PATH"
        if os.environ.get("UNDISTORTME_REQUIRE_FSL"):
            pytest.fail(msg)
        pytest.skip(msg)

    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI")
    run_info = _make_smoke_bids_run(tmp_path)

    check_dict = {
        "dcm2niix": False,
        "topup": True,
        "topup_multithread": False,
        "slice": True,
        "match": False,
        "dryrun": False,
        "two_echo": False,
        "mask": False,
        "jobs": 4,
        "oversubscribe": 1.0,
    }
    monkeypatch.setattr(cp, "check_dict", check_dict, raising=False)
    monkeypatch.setattr(cp, "failed_commands", [])

    perslice_cnf = str(_REPO_ROOT / "configs" / "perslice_1.cnf")
    cp.process_run(
        run_info["subject"],
        run_info["session"],
        run_info["run"],
        run_info["output_dir"],
        run_info["deriv_dir"],
        perslice_cnf,
    )

    assert cp.failed_commands == [], (
        f"{len(cp.failed_commands)} command(s) failed, first: "
        f"{cp.failed_commands[0]}"
    )
    results_dir = Path(run_info["deriv_dir"]) / "undistortme" / "per-slice" / \
        run_info["subject"] / run_info["session"] / run_info["run"]
    recombined = sorted(glob.glob(str(results_dir / "*recombined*")))
    assert len(recombined) == 2, (
        f"expected 2 recombined volumes (one per echo) in {results_dir}, "
        f"found {len(recombined)}"
    )
    for path in recombined:
        img = nib.load(path)
        assert img.shape == (32, 32, 16), \
            f"recombined shape {img.shape} != input shape (32, 32, 16): {path}"
        assert np.isfinite(img.get_fdata()).all()
