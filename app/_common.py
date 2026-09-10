"""_common.py
Shared Streamlit UI helpers for app/setup_dataset.py (and anything else
under app/ that wants them) -- native path pickers, a live-streaming
progress console, and image-to-data-URL encoding for st.column_config.
ImageColumn (which only accepts URLs/data-URLs, never local file paths).

Not imported by any CLI tool under src/ -- this is presentation-only code,
kept out of the core/reusable modules on purpose (see CONVENTIONS.md
"Fonctions core reutilisables").
"""
from __future__ import annotations

import base64
import contextlib
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

LOGO_PATH = Path("app/assets/idmb_logo.png")


def discover(pattern: str) -> list[str]:
    return sorted(str(p) for p in Path(".").glob(pattern))


def save_uploaded_files(uploaded_files, dest_dir: Path) -> list[Path]:
    """Writes one or more st.file_uploader UploadedFile objects to
    `dest_dir` under their original names and returns the resulting paths
    -- st.file_uploader only hands back in-memory bytes, and every
    file-based reader in src/ (core.tps_io.parse_tps, pd.read_csv(path),
    ...) needs a real path on disk, so a caller that wants to reuse those
    readers has to persist the upload first (into a throwaway directory it
    owns, e.g. a tempfile.TemporaryDirectory -- this function doesn't
    clean up after itself)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for uploaded_file in uploaded_files:
        dest = dest_dir / uploaded_file.name
        dest.write_bytes(uploaded_file.getvalue())
        paths.append(dest)
    return paths


def _session_upload_dir() -> Path:
    """A scratch dir that outlives a single script rerun, unlike a `with
    tempfile.TemporaryDirectory()` block -- created once per browser
    session (mkdtemp, never cleaned up here) and reused by every
    `file_picker()` call in that session. Needed by a multi-step wizard
    (app/setup_dataset.py) where an uploaded file's path must still
    resolve several reruns/steps later, not just within the run that
    uploaded it."""
    if "_common_upload_dir" not in st.session_state:
        st.session_state["_common_upload_dir"] = tempfile.mkdtemp(prefix="idmybee_upload_")
    return Path(st.session_state["_common_upload_dir"])


def file_picker(
    label: str, *, key: str, help: str | None = None, type: str | list[str] | None = None,
) -> str:
    """st.file_uploader wrapper returning a real path on disk -- same
    calling convention as path_picker(mode="file"): a path string once
    something's uploaded, "" until then. Every file-based reader in src/
    (pd.read_csv(path), open(path), a CLI --flag, ...) needs a real path;
    st.file_uploader only ever hands back in-memory bytes (see
    save_uploaded_files/_session_upload_dir). Use path_picker(mode="dir")
    instead for a folder -- there's no file to upload, just a location to
    point at."""
    uploaded = st.file_uploader(label, key=key, help=help, type=type)
    if uploaded is None:
        return ""
    return str(save_uploaded_files([uploaded], _session_upload_dir())[0])


# ---------------------------------------------------------------------------
# Native OS path picker
# ---------------------------------------------------------------------------

_PICKER_SCRIPT = """
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
if sys.argv[1] == "dir":
    path = filedialog.askdirectory(initialdir=sys.argv[2] or None)
elif sys.argv[1] == "save":
    initial_dir = sys.argv[2] or None
    initial_file = sys.argv[3] if len(sys.argv) > 3 else ""
    path = filedialog.asksaveasfilename(initialdir=initial_dir, initialfile=initial_file)
else:
    path = filedialog.askopenfilename(initialdir=sys.argv[2] or None)
print(path)
"""


def _browse(mode: str, initial_dir: str, initial_file: str = "") -> str | None:
    """Opens a native folder/file picker via a throwaway subprocess -- Tk
    must own the main thread of its own process, and Streamlit's script
    thread isn't the main thread of this one (an in-process tkinter call
    hangs/crashes, notably on macOS). Local-machine-only affordance: opens
    a window on whichever machine runs `streamlit run`, same assumption
    the app's own docstring already makes. Returns None if the user
    cancelled or no display is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PICKER_SCRIPT, mode, initial_dir, initial_file],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    path = result.stdout.strip()
    return path or None


def path_picker(
    label: str, *, mode: str = "dir", value: str = "", key: str, help: str | None = None,
    default_filename: str = "",
) -> str:
    """A text input plus a "Browse..." button that opens the OS's native
    folder/file/save picker (mode="dir"/"file"/"save") and writes the
    chosen path back into the text input. mode="save" (filedialog.
    asksaveasfilename, pre-filled with `default_filename`) is for a path
    that doesn't exist yet (e.g. an optional predictions.csv export
    location), as opposed to mode="file" which is for picking an existing
    file. Returns the current text value (typed or browsed) every rerun,
    same calling convention as st.text_input.

    Once a widget has its own `key`, Streamlit always displays that
    widget's own persisted state on rerun and ignores `value` -- so two
    things need care here: writing a Browse result back has to happen
    *before* the text_input with that key is instantiated in the next run
    (assigning to a widget's own session_state key after it's already been
    instantiated this run raises), and a caller-supplied `value` has to be
    re-applied only when it actually changed (otherwise it would stomp on
    whatever the user just typed on every single rerun)."""
    widget_key = f"{key}__widget"
    pending_key = f"{key}__pending"
    seen_key = f"{key}__seen_value"

    if widget_key not in st.session_state:
        st.session_state[widget_key] = value
        st.session_state[seen_key] = value
    elif value != st.session_state.get(seen_key):
        st.session_state[widget_key] = value
        st.session_state[seen_key] = value

    if pending_key in st.session_state:
        st.session_state[widget_key] = st.session_state.pop(pending_key)

    with st.container(horizontal=True):
        text_value = st.text_input(label, help=help, key=widget_key)
        if st.button(":material/folder_open: Parcourir...", key=f"{key}__browse"):
            initial_dir = text_value if mode == "dir" and text_value else (str(Path(text_value).parent) if text_value else "")
            chosen = _browse(mode, initial_dir, default_filename if mode == "save" else "")
            if chosen:
                st.session_state[pending_key] = chosen
                st.rerun()
    return text_value


# ---------------------------------------------------------------------------
# Live-streaming run log + progress bar
# ---------------------------------------------------------------------------

# Two conventions every batch-processing stage under src/ already prints on
# stdout (see e.g. extraction/detect_wing.py, extraction/normalize_crop.py,
# landmarks/predict.py) -- parsed here rather than threading a progress
# callback through every CLI main(argv), so this stays presentation-only
# (see module docstring) and every stage keeps working identically from a
# plain terminal:
#   "=== Section name ==="       -- a sub-operation starting (e.g. one of
#                                    utils.landmarking_pipeline.run_detection_
#                                    and_crop's two stages)
#   "[done/total] ...detail..."  -- a batch checkpoint within it
_SECTION_RE = re.compile(r"^=+\s*(.+?)\s*=+$")
_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]\s*(.*)$")


class _LiveBuffer(io.StringIO):
    """A stdout target that mirrors every write() into a placeholder
    immediately -- fn() runs synchronously in the same script execution as
    this call, so updating a Streamlit element mid-call is enough to
    stream output live, no threading/queueing needed. Also drives a
    progress bar (see _SECTION_RE/_PROGRESS_RE above) when the caller
    passes one -- stays untouched (never rendered) for a stage that prints
    neither convention, e.g. classifiers.train's single fit/LOOCV call."""

    def __init__(self, placeholder, progress_placeholder=None):
        super().__init__()
        self._placeholder = placeholder
        self._progress_placeholder = progress_placeholder
        self._pending_line = ""
        self._section = ""

    def write(self, s: str) -> int:
        n = super().write(s)
        if s:
            self._placeholder.code(self.getvalue(), language=None)
            if self._progress_placeholder is not None:
                self._scan_for_progress(s)
        return n

    def _scan_for_progress(self, chunk: str) -> None:
        """Buffers a partial last line across write() calls (print()'s own
        chunking is not guaranteed to land on newline boundaries) and hands
        each complete line to _handle_line()."""
        self._pending_line += chunk
        *complete_lines, self._pending_line = self._pending_line.split("\n")
        for line in complete_lines:
            self._handle_line(line.strip())

    def _handle_line(self, line: str) -> None:
        section_match = _SECTION_RE.match(line)
        if section_match:
            self._section = section_match.group(1)
            self._progress_placeholder.progress(0.0, text=self._section)
            return

        progress_match = _PROGRESS_RE.match(line)
        if progress_match:
            done, total, detail = progress_match.groups()
            done, total = int(done), int(total)
            fraction = min(done / total, 1.0) if total else 1.0
            label = f"{self._section} -- {done}/{total}" if self._section else f"{done}/{total}"
            if detail:
                label += f" ({detail})"
            self._progress_placeholder.progress(fraction, text=label)


def run_with_log(label: str, fn, *args, **kwargs):
    """Runs a pipeline stage inside a live-updating st.status console:
    every print() the stage's own main()/run_batch() makes (see
    CONVENTIONS.md "Logging") appears in the console as it happens,
    instead of only after the whole stage finishes -- plus a progress bar
    above it, parsed from that same stdout (see _LiveBuffer), for whichever
    stage actually reports batch progress that way. logger.info/warning
    messages still go only to the terminal running `streamlit run`, same
    as before -- only print() output is captured here."""
    status = st.status(label, expanded=True)
    with status:
        progress_placeholder = st.empty()
        placeholder = st.empty()
        placeholder.code("(nothing printed yet)", language=None)
        buffer = _LiveBuffer(placeholder, progress_placeholder)
        try:
            with contextlib.redirect_stdout(buffer):
                result = fn(*args, **kwargs)
        except BaseException:
            # BaseException, not Exception: several of the wrapped CLI
            # main()s raise SystemExit on a user-facing error (e.g. a bad
            # --mode/--heavy-ref combination), which doesn't subclass
            # Exception -- still worth flagging the console as failed
            # (and keeping it expanded) before propagating.
            status.update(label=f"{label} -- failed", state="error", expanded=True)
            raise
    status.update(label=label, state="complete", expanded=False)
    return result


# ---------------------------------------------------------------------------
# Local images -> data URLs, for st.column_config.ImageColumn
# ---------------------------------------------------------------------------

def _encode_bgr(image: np.ndarray, max_width: int) -> str:
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


@st.cache_data(max_entries=4096, show_spinner=False)
def _image_path_to_data_url(path: str, mtime: float, max_width: int) -> str:
    """Cached on (path, mtime) so a photo already reviewed once doesn't get
    re-read/re-encoded on every table rerun (data_editor reruns the whole
    script on every cell edit)."""
    image = cv2.imread(path)
    if image is None:
        return ""
    return _encode_bgr(image, max_width)


def image_to_data_url(path: str, *, max_width: int = 360) -> str:
    """Data URL for a crop already on disk at `path` -- empty string if
    missing/unreadable (caller shows a placeholder for that)."""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return _image_path_to_data_url(str(file_path), file_path.stat().st_mtime, max_width)


def array_to_data_url(image: np.ndarray, *, max_width: int = 360) -> str:
    """Data URL for an in-memory BGR image (e.g. a landmark overlay built
    on the fly) -- not cached here, caller decides the cache key since the
    inputs aren't hashable/stable the way a file path+mtime is."""
    return _encode_bgr(image, max_width)


# ---------------------------------------------------------------------------
# Reference shape preview
# ---------------------------------------------------------------------------

def plot_reference_shape(zones: np.ndarray):
    """Numbered scatter of a reference shape's (x, y) zones (see
    landmarks.build_reference/landmarks.renumber) -- lets a non-developer
    eyeball a reference .tps (picked to land landmarking on, or just
    built) before relying on it, instead of trusting an opaque file. Y is
    flipped since these are image-space coordinates (origin top-left)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(zones[:, 0], zones[:, 1], s=24)
    for i, (x, y) in enumerate(zones):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig
