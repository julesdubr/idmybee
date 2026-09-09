"""_common.py
Shared Streamlit UI helpers for app/build_dataset.py (and anything else
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
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

LOGO_PATH = Path("app/assets/idmb_logo.png")


def discover(pattern: str) -> list[str]:
    return sorted(str(p) for p in Path(".").glob(pattern))


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
else:
    path = filedialog.askopenfilename(initialdir=sys.argv[2] or None)
print(path)
"""


def _browse(mode: str, initial_dir: str) -> str | None:
    """Opens a native folder/file picker via a throwaway subprocess -- Tk
    must own the main thread of its own process, and Streamlit's script
    thread isn't the main thread of this one (an in-process tkinter call
    hangs/crashes, notably on macOS). Local-machine-only affordance: opens
    a window on whichever machine runs `streamlit run`, same assumption
    the app's own docstring already makes. Returns None if the user
    cancelled or no display is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PICKER_SCRIPT, mode, initial_dir],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    path = result.stdout.strip()
    return path or None


def path_picker(
    label: str, *, mode: str = "dir", value: str = "", key: str, help: str | None = None,
) -> str:
    """A text input plus a "Browse..." button that opens the OS's native
    folder/file picker (mode="dir"/"file") and writes the chosen path back
    into the text input. Returns the current text value (typed or
    browsed) every rerun, same calling convention as st.text_input.

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
        if st.button(":material/folder_open: Browse...", key=f"{key}__browse"):
            initial_dir = text_value if mode == "dir" and text_value else (str(Path(text_value).parent) if text_value else "")
            chosen = _browse(mode, initial_dir)
            if chosen:
                st.session_state[pending_key] = chosen
                st.rerun()
    return text_value


# ---------------------------------------------------------------------------
# Live-streaming run log
# ---------------------------------------------------------------------------

class _LiveBuffer(io.StringIO):
    """A stdout target that mirrors every write() into a placeholder
    immediately -- fn() runs synchronously in the same script execution as
    this call, so updating a Streamlit element mid-call is enough to
    stream output live, no threading/queueing needed."""

    def __init__(self, placeholder):
        super().__init__()
        self._placeholder = placeholder

    def write(self, s: str) -> int:
        n = super().write(s)
        if s:
            self._placeholder.code(self.getvalue(), language=None)
        return n


def run_with_log(label: str, fn, *args, **kwargs):
    """Runs a pipeline stage inside a live-updating st.status console:
    every print() the stage's own main()/run_batch() makes (see
    CONVENTIONS.md "Logging") appears in the console as it happens,
    instead of only after the whole stage finishes. logger.info/warning
    messages still go only to the terminal running `streamlit run`, same
    as before -- only print() output is captured here."""
    status = st.status(label, expanded=True)
    with status:
        placeholder = st.empty()
        placeholder.code("(nothing printed yet)", language=None)
        buffer = _LiveBuffer(placeholder)
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
