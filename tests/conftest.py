import difflib
import json
import os
import re
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest
from undistortme import pipeline as cp


@pytest.fixture
def check_dict(monkeypatch):
    """Install a fresh all-False gate dict as the module-global corr_pipeline.check_dict.

    The real dict is created only inside main(), so the module attribute does not
    exist at import time.  monkeypatch.setattr with raising=False creates it and
    automatically deletes it again on teardown.
    """
    d = {
        "dcm2niix": False,
        "topup": False,
        "topup_multithread": False,
        "slice": False,
        "match": False,
        "dryrun": False,
        "two_echo": False,
        "mask": False,
    }
    monkeypatch.setattr(cp, "check_dict", d, raising=False)
    return d


@pytest.fixture
def tiny_nii():
    """Return a factory that writes a real uncompressed NIfTI and returns the path str.

    Usage:
        path = tiny_nii(tmp_path / "img.nii")
        path = tiny_nii(tmp_path / "img.nii", shape=(4, 4, 4), fill=1.0)
        path = tiny_nii(tmp_path / "img.nii", seed=42)

    Default shape (8, 8, 8) is large enough for skimage SSIM's default win_size.
    """

    def _make(path, shape=(8, 8, 8), fill=None, seed=0):
        if fill is not None:
            data = np.full(shape, float(fill), dtype=np.float32)
        else:
            data = np.random.default_rng(seed).random(shape, dtype=np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
        return str(path)

    return _make


# ===========================================================================
# Orchestration-test harness (Task 5)
#
# These fixtures/helpers support tests/test_process_run_orchestration.py.  They
# let process_run run end-to-end WITHOUT executing any external binary: every
# shell command the pipeline would dispatch is captured at the three batch
# dispatchers (parallel_bash_commands / serial_bash_commands) and
# recorded instead of run.
# ===========================================================================

# The contrast-match subcommand baked into the pipeline's commands.
CMATCH_PATH = cp.CONTRASTMATCH_CMD


def _make_bids_run(tmp_path, tiny_nii, *, n_echoes=3,
                   te=(0.07, 0.09, 0.11, 0.13), phase_dir="j-", bvals=None,
                   n_avgs=1, subject="sub-01", session="ses-01", run="run-01",
                   output_root=None, deriv_root=None):
    """Build a BIDS-ish run tree the pipeline can glob and pin.

    Layout produced under ``{output_root}/{subject}/{session}/{run}/``:

      * sidecar  ``{subject}_{session}_{run}_echo-{e}.json``  (e = 1..n_echoes)
      * volumes  ``{subject}_{session}_{run}_echo-{e}_{bb}.nii``
        with ``bb = str(i).zfill(len(str(N)))``, i = 1..N where
        N = len(bvals) when bvals is given else n_avgs.
      * (optional) ``{subject}_{session}_{run}_echo-{e}.bval`` containing the
        space-separated bvals (one identical .bval per echo).

    All .nii are real (tiny, 8x8x8, seeded) so that code paths which call
    ``nib.load`` (run_topup_diffusion_special / find_closest_volume_nmi) work.

    Returns a dict with output_dir, deriv_dir, subject, session, run, run_dir,
    tmp_path (all strings).
    """
    out = Path(output_root) if output_root else (tmp_path / "out")
    deriv = Path(deriv_root) if deriv_root else (tmp_path / "deriv")
    run_dir = out / subject / session / run
    run_dir.mkdir(parents=True, exist_ok=True)

    n_vols = len(bvals) if bvals is not None else n_avgs
    num_digits = len(str(n_vols))

    for e in range(1, n_echoes + 1):
        stem = f"{subject}_{session}_{run}_echo-{e}"
        data = {
            "EchoNumber": e,
            "EchoTime": te[e - 1],
            "PhaseEncodingDirection": phase_dir,
            "TotalReadoutTime": 0.106487,
            "PulseSequenceName": "ep_seg_35",
            "SeriesDescription": "ME_GRE",
            "ImageType": ["ORIGINAL", "PRIMARY", "M"],
            "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
        }
        (run_dir / f"{stem}.json").write_text(json.dumps(data))
        if bvals is not None:
            (run_dir / f"{stem}.bval").write_text(
                " ".join(str(b) for b in bvals))
        for i in range(1, n_vols + 1):
            bb = str(i).zfill(num_digits)
            tiny_nii(run_dir / f"{stem}_{bb}.nii", seed=(e * 100 + i))

    return {
        "output_dir": str(out),
        "deriv_dir": str(deriv),
        "subject": subject,
        "session": session,
        "run": run,
        "run_dir": str(run_dir),
        "tmp_path": str(tmp_path),
    }


@pytest.fixture
def bids_run(tmp_path, tiny_nii):
    """Factory building a BIDS run tree (see _make_bids_run for the contract)."""

    def _build(**kwargs):
        return _make_bids_run(tmp_path, tiny_nii, **kwargs)

    return _build


def _clean(commands):
    """Match dispatcher behavior: drop None entries from a command list."""
    if commands is None:
        return []
    return [c for c in commands if c is not None]


def _install_recorder(monkeypatch, record_fn):
    """Patch the three batch dispatchers + shuffle to use ``record_fn``.

    All three dispatchers share the signature ``(bash_commands, description)``.
    ``shuffle`` is no-op'd so work lists keep deterministic (source) order for
    stable snapshots.  ``record_fn`` receives ``(bash_commands, description)``
    and is responsible for building/appending to whatever call log it owns.
    """
    monkeypatch.setattr(cp, "parallel_bash_commands", record_fn)
    monkeypatch.setattr(cp, "serial_bash_commands", record_fn)
    monkeypatch.setattr(cp, "shuffle", lambda seq: None)


@pytest.fixture
def recorder(monkeypatch):
    """Capture every dispatched batch instead of executing it.

    Monkeypatches the batch dispatchers on the pipeline module so
    that each call appends ``(description, [non-None commands])`` to the
    returned list, executing nothing.  Also no-ops ``cp.shuffle`` so that work
    lists keep deterministic (source) order for stable snapshots.

    All three dispatchers share the signature ``(bash_commands, description)``.
    """
    calls = []

    def _record(bash_commands, description):
        calls.append((description, _clean(bash_commands)))

    _install_recorder(monkeypatch, _record)
    return calls


@pytest.fixture
def slicing_recorder(monkeypatch, tiny_nii):
    """Like ``recorder`` but also fakes the on-disk outputs the pipeline reads.

    The whole-volume paths never touch disk for their outputs, but two batches
    have outputs that LATER code globs/reads, so we must materialise them:

      * slicenii batch (``slicenii -i {nii} -o {slice_dir} -p 6``): the real
        slicenii would write
        ``{slice_dir}/{base}_slices/{base}_axis-2_slice-padded-{NNN}.nii``
        (base = basename(nii) without extension, NNN 3-digit 1-based).
        handle_slicing then globs ``{slice_dir}/{base}_slices/{base}_*`` and
        parses the slice number from the filename, so we write 3 such slices
        per input volume (real tiny niftis).
      * masking batch (``fslmaths {nii} -mas {mask} {out}``): process_run then
        rewrites run_df["nii"] to the masked outputs.  We write each ``{out}``
        (the last token) as a real tiny nifti so downstream steps can proceed.

    Parsing is intentionally literal (token splitting on the exact command
    shapes above); anything else is recorded but not faked.
    """
    calls = []

    def _fake_outputs(cmd):
        tokens = cmd.split()
        if cmd.startswith("slicenii "):
            nii_path = tokens[tokens.index("-i") + 1]
            slice_dir = tokens[tokens.index("-o") + 1]
            base = os.path.basename(nii_path).split(".")[0]
            out_dir = Path(slice_dir) / f"{base}_slices"
            out_dir.mkdir(parents=True, exist_ok=True)
            for n in range(1, 4):
                fname = f"{base}_axis-2_slice-padded-{str(n).zfill(3)}.nii"
                tiny_nii(out_dir / fname, seed=1000 + n)
        elif cmd.startswith("fslmaths ") and " -mas " in cmd:
            # tokens[-1] is assumed to be the output path, mirroring the current
            # "fslmaths {nii} -mas {mask} {out}" command shape.  If the masking
            # command shape changes (extra flags, reordered args, etc.) this
            # assumption will silently miss the real output — revisit then.
            out_path = tokens[-1]
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            tiny_nii(out_path, seed=2000)

    def _record(bash_commands, description):
        cmds = _clean(bash_commands)
        for cmd in cmds:
            _fake_outputs(cmd)
        calls.append((description, cmds))

    _install_recorder(monkeypatch, _record)
    return calls


def _normalize(calls, replacements):
    """Render captured (description, commands) batches to snapshot text.

    For each batch emit ``## {description}`` then one command per line, in the
    captured order (NOT sorted: order is characterized behavior).  Path
    substitutions are applied longest-find-first so that nested roots (deriv /
    output dirs under tmp_path) win over their tmp_path prefix.  The hardcoded
    contrastmatch.py path is always mapped to ``<CMATCH>``, and a
    ``--nthr=<N>`` backstop guards against any leaked cpu-count.

    Compound commands joined by ``" && "`` are split so that each subcommand
    appears on its own continuation line (indented 4 spaces after the first),
    i.e. ``" && "`` is replaced with ``" &&\\n    "``.  This makes line-granular
    diffs point at the exact subcommand that changed rather than a single
    multi-kilobyte line.

    ``replacements`` is an iterable of ``(find, placeholder)`` pairs.
    """
    reps = list(replacements) + [(CMATCH_PATH, "<CMATCH>")]
    reps.sort(key=lambda kv: len(kv[0]), reverse=True)

    lines = []
    for description, commands in calls:
        lines.append(f"## {description}")
        for cmd in commands:
            s = cmd
            for find, placeholder in reps:
                s = s.replace(find, placeholder)
            s = re.sub(r"--nthr=\d+", "--nthr=<N>", s)
            s = s.replace(" && ", " &&\n    ")
            lines.append(s)
    return "\n".join(lines) + "\n"


@pytest.fixture
def normalize():
    """Return the ``_normalize(calls, replacements)`` snapshot renderer."""
    return _normalize


_SNAPSHOT_DIR = Path(__file__).parent / "_snapshots"


def _assert_snapshot(name, text):
    """Compare ``text`` against the committed golden snapshot ``{name}.txt``.

    With env ``UPDATE_SNAPSHOTS`` set, (over)write the golden and skip.  Without
    it, fail with a unified diff on mismatch (or if the golden is missing).
    """
    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = _SNAPSHOT_DIR / f"{name}.txt"

    if os.environ.get("UPDATE_SNAPSHOTS", "").lower() in {"1", "true", "yes"}:
        path.write_text(text)
        pytest.skip(f"snapshot written: {path.name}")

    if not path.exists():
        pytest.fail(
            f"Snapshot {path.name} missing. "
            f"Regenerate with UPDATE_SNAPSHOTS=1.")

    expected = path.read_text()
    if text != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile=f"golden/{path.name}",
                tofile="actual",
            ))
        pytest.fail(f"Snapshot mismatch for {path.name}:\n{diff}")


@pytest.fixture
def assert_snapshot():
    """Return the ``_assert_snapshot(name, text)`` comparator."""
    return _assert_snapshot
