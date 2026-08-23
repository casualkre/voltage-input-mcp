"""Streaming capture: xdg-desktop-portal ScreenCast -> PipeWire -> GStreamer appsink.

Per-call screenshots are fine at 2 Hz but they are the wrong shape for this project.
Probes and reflexes want frames as fast as the compositor produces them, and paying a
fresh compositor round trip for each one wastes most of the budget. A PipeWire stream
gives a continuous feed after a single setup cost, so `grab()` becomes "read the newest
frame from a slot" -- microseconds, not milliseconds.

The cost is a permission dialog. The portal shows a picker the first time; passing
`persist_mode=2` and storing the returned restore token means subsequent runs reuse the
grant silently. The token is written to the state directory and reused automatically.

Portal calls are asynchronous in an awkward way: each method returns a Request object
path and the real answer arrives later as a `Response` signal on that path. That needs a
running GLib main loop, which lives on a dedicated thread here.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import CaptureError
from .base import CaptureBackend, Frame

__all__ = ["PortalBackend", "available"]

_PORTAL_BUS = "org.freedesktop.portal.Desktop"
_PORTAL_PATH = "/org/freedesktop/portal/desktop"
_SCREENCAST = "org.freedesktop.portal.ScreenCast"
_REQUEST_IFACE = "org.freedesktop.portal.Request"

SOURCE_MONITOR = 1
SOURCE_WINDOW = 2

CURSOR_HIDDEN = 1
CURSOR_EMBEDDED = 2
CURSOR_METADATA = 4

PERSIST_NONE = 0
PERSIST_TRANSIENT = 1
PERSIST_PERSISTENT = 2


def available() -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gio, Gst  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    try:
        from gi.repository import Gst

        if not Gst.is_initialized():
            Gst.init(None)
        return Gst.ElementFactory.find("pipewiresrc") is not None
    except Exception:  # noqa: BLE001
        return False


class _MainLoopThread:
    """A GLib main loop on its own thread, shared by all portal sessions.

    This deliberately drives the **global default** MainContext rather than a private
    one. The DBus connection is created with `Gio.bus_get_sync` on whichever thread got
    there first, which binds its signal delivery to the thread-default context in effect
    at that moment -- the global default. Running a private context here would leave
    portal `Response` signals queued on the default context with nothing iterating it,
    and every portal call would time out waiting for a reply that had already arrived.
    """

    _instance: _MainLoopThread | None = None
    _guard = threading.Lock()

    def __init__(self) -> None:
        from gi.repository import GLib

        self.loop = GLib.MainLoop.new(None, False)
        self._thread = threading.Thread(target=self._run, name="voltage-glib", daemon=True)
        self._ready = threading.Event()
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        self._ready.set()
        self.loop.run()

    @classmethod
    def get(cls) -> _MainLoopThread:
        with cls._guard:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance


class PortalBackend(CaptureBackend):
    """Continuous screen capture via the ScreenCast portal."""

    name = "portal"

    def __init__(
        self,
        *,
        state_dir: Path | None = None,
        cursor_mode: int = CURSOR_EMBEDDED,
        source_types: int = SOURCE_MONITOR,
        setup_timeout_s: float = 90.0,
    ) -> None:
        self._state_dir = state_dir or Path(
            os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")
        ) / "voltage-input-mcp"
        self._cursor_mode = cursor_mode
        self._source_types = source_types
        self._setup_timeout = setup_timeout_s

        self._conn: Any = None
        self._session: str | None = None
        self._pipeline: Any = None
        self._appsink: Any = None
        self._stream_origin: tuple[int, int] = (0, 0)

        self._slot_lock = threading.Lock()
        self._latest: Frame | None = None
        self._frame_id = 0
        self._first_frame = threading.Event()
        self._started = False

    # -- lifecycle -------------------------------------------------------------------

    @property
    def streaming(self) -> bool:
        return True

    def start(self) -> None:
        if self._started:
            return
        _MainLoopThread.get()  # ensure signal delivery before any portal call
        self._connect()
        node_id, origin = self._negotiate()
        self._stream_origin = origin
        fd = self._open_remote()
        self._build_pipeline(fd, node_id)
        if not self._first_frame.wait(timeout=10.0):
            self.stop()
            raise CaptureError(
                "PipeWire stream produced no frames within 10s. The portal grant may "
                "have selected a source that is not currently rendering."
            )
        self._started = True

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                from gi.repository import Gst

                self._pipeline.set_state(Gst.State.NULL)
            except Exception:  # noqa: BLE001
                pass
            self._pipeline = None
            self._appsink = None
        if self._session and self._conn is not None:
            try:
                self._conn.call_sync(
                    _PORTAL_BUS, self._session, "org.freedesktop.portal.Session",
                    "Close", None, None, _flags(), 2000, None,
                )
            except Exception:  # noqa: BLE001
                pass
        self._session = None
        self._started = False
        self._first_frame.clear()

    # -- capture ---------------------------------------------------------------------

    def grab(self, region: tuple[int, int, int, int] | None = None) -> Frame:
        if not self._started:
            self.start()
        with self._slot_lock:
            frame = self._latest
        if frame is None:
            raise CaptureError("no frame available from the PipeWire stream yet")
        if region is not None:
            x, y, w, h = region
            ox, oy = frame.origin
            return frame.crop(x - ox, y - oy, w, h)
        return frame

    def health(self) -> dict[str, object]:
        """Report readiness without provoking a permission dialog.

        The base implementation captures a frame, which for this backend means running
        the full portal handshake and showing the user a screen-share picker. A
        diagnostic command must not do that as a side effect. If a restore token exists
        the grant is already given and starting is silent, so a real capture is fine;
        otherwise report that the first capture will prompt, and say so plainly.
        """
        if self._started:
            return super().health()
        if not self._load_restore_token():
            return {
                "backend": self.name,
                "ok": True,
                "streaming": True,
                "started": False,
                "note": (
                    "available, but not yet authorised. The first capture shows a "
                    "one-time screen-share picker; the grant is then persisted and "
                    "reused silently on later runs."
                ),
            }
        return super().health()

    # -- portal handshake ------------------------------------------------------------

    def _connect(self) -> None:
        from gi.repository import Gio

        if self._conn is None:
            try:
                self._conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            except Exception as exc:  # noqa: BLE001
                raise CaptureError(f"cannot reach the session bus: {exc}") from exc

    def _token(self, prefix: str) -> str:
        # Must be a valid DBus path element: letters, digits and underscore only.
        return f"voltage_{prefix}_{os.getpid()}_{int(time.monotonic() * 1000) % 1_000_000}"

    def _request_path(self, token: str) -> str:
        sender = self._conn.get_unique_name()[1:].replace(".", "_")
        return f"/org/freedesktop/portal/desktop/request/{sender}/{token}"

    def _call_and_wait(self, method: str, args: tuple, token: str) -> dict[str, Any]:
        """Invoke a portal method and block until its Request emits Response."""
        from gi.repository import GLib

        done = threading.Event()
        box: dict[str, Any] = {}
        _MainLoopThread.get()
        path = self._request_path(token)

        def on_response(_conn, _sender, _path, _iface, _signal, params) -> None:
            code, results = params.unpack()
            box["code"] = code
            box["results"] = results
            done.set()

        sub = self._conn.signal_subscribe(
            _PORTAL_BUS, _REQUEST_IFACE, "Response", path, None,
            _sig_flags(), on_response,
        )
        try:
            self._conn.call_sync(
                _PORTAL_BUS, _PORTAL_PATH, _SCREENCAST, method,
                GLib.Variant(*args), GLib.VariantType("(o)"), _flags(),
                int(self._setup_timeout * 1000), None,
            )
            if not done.wait(timeout=self._setup_timeout):
                raise CaptureError(
                    f"portal {method} timed out after {self._setup_timeout:.0f}s. "
                    "The screen-share dialog may be waiting for a response."
                )
        finally:
            self._conn.signal_unsubscribe(sub)

        code = box.get("code", 2)
        if code != 0:
            reason = {1: "cancelled by the user", 2: "ended unexpectedly"}.get(code, str(code))
            raise CaptureError(f"portal {method} was refused: {reason}")
        return box.get("results") or {}

    def _negotiate(self) -> tuple[int, tuple[int, int]]:
        from gi.repository import GLib

        session_token = self._token("session")
        create_token = self._token("create")
        results = self._call_and_wait(
            "CreateSession",
            (
                "(a{sv})",
                ({
                    "handle_token": GLib.Variant("s", create_token),
                    "session_handle_token": GLib.Variant("s", session_token),
                },),
            ),
            create_token,
        )
        session = results.get("session_handle")
        if not session:
            raise CaptureError("portal CreateSession returned no session handle")
        self._session = session

        select_token = self._token("select")
        options: dict[str, Any] = {
            "handle_token": GLib.Variant("s", select_token),
            "types": GLib.Variant("u", self._source_types),
            "multiple": GLib.Variant("b", False),
            "cursor_mode": GLib.Variant("u", self._cursor_mode),
            "persist_mode": GLib.Variant("u", PERSIST_PERSISTENT),
        }
        restore = self._load_restore_token()
        if restore:
            options["restore_token"] = GLib.Variant("s", restore)

        self._call_and_wait(
            "SelectSources", ("(oa{sv})", (session, options)), select_token
        )

        start_token = self._token("start")
        results = self._call_and_wait(
            "Start",
            ("(osa{sv})", (session, "", {"handle_token": GLib.Variant("s", start_token)})),
            start_token,
        )

        if new_token := results.get("restore_token"):
            self._save_restore_token(str(new_token))

        streams = results.get("streams") or []
        if not streams:
            raise CaptureError("portal Start returned no streams")
        node_id, props = streams[0]
        position = props.get("position") or (0, 0)
        return int(node_id), (int(position[0]), int(position[1]))

    def _open_remote(self) -> int:
        from gi.repository import Gio, GLib

        try:
            reply, fd_list = self._conn.call_with_unix_fd_list_sync(
                _PORTAL_BUS, _PORTAL_PATH, _SCREENCAST, "OpenPipeWireRemote",
                GLib.Variant("(oa{sv})", (self._session, {})),
                GLib.VariantType("(h)"), Gio.DBusCallFlags.NONE, 10_000, None, None,
            )
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(f"OpenPipeWireRemote failed: {exc}") from exc
        index = reply.unpack()[0]
        return fd_list.get(index)

    # -- GStreamer -------------------------------------------------------------------

    def _build_pipeline(self, fd: int, node_id: int) -> None:
        from gi.repository import Gst

        if not Gst.is_initialized():
            Gst.init(None)

        # always-copy=false would hand us mapped DMA-BUF memory we cannot read from the
        # CPU without an import step; copying is the simple, portable choice and the
        # copy is dwarfed by the videoconvert that follows anyway.
        desc = (
            f"pipewiresrc fd={fd} path={node_id} always-copy=true do-timestamp=true "
            f"! videoconvert ! video/x-raw,format=RGB "
            f"! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        try:
            pipeline = Gst.parse_launch(desc)
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(f"could not build the GStreamer pipeline: {exc}") from exc

        appsink = pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_sample)

        self._pipeline = pipeline
        self._appsink = appsink
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._pipeline = None
            raise CaptureError("GStreamer refused to start the PipeWire pipeline")

    def _on_sample(self, sink) -> Any:
        from gi.repository import Gst

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        try:
            frame = self._sample_to_frame(sample)
        except Exception:  # noqa: BLE001 - a bad frame must not kill the stream
            return Gst.FlowReturn.OK
        with self._slot_lock:
            self._latest = frame
        self._first_frame.set()
        return Gst.FlowReturn.OK

    def _sample_to_frame(self, sample) -> Frame:
        from gi.repository import Gst

        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")
        buf = sample.get_buffer()

        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise CaptureError("could not map the GStreamer buffer")
        try:
            data = bytes(info.data)
        finally:
            buf.unmap(info)

        # GStreamer pads each row up to a 4-byte boundary for RGB.
        stride = len(data) // height if height else width * 3
        arr = np.frombuffer(data[: stride * height], dtype=np.uint8).reshape(height, stride)
        pixels = np.ascontiguousarray(arr[:, : width * 3].reshape(height, width, 3))

        self._frame_id += 1
        return Frame(
            pixels=pixels, origin=self._stream_origin, ts=time.monotonic(),
            frame_id=self._frame_id, backend=self.name,
        )

    # -- restore token ---------------------------------------------------------------

    def _token_file(self) -> Path:
        return self._state_dir / "portal-restore.json"

    def _load_restore_token(self) -> str | None:
        try:
            data = json.loads(self._token_file().read_text())
            token = data.get("restore_token")
            return str(token) if token else None
        except (OSError, ValueError):
            return None

    def _save_restore_token(self, token: str) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            path = self._token_file()
            path.write_text(json.dumps({"restore_token": token}, indent=2))
            path.chmod(0o600)
        except OSError:
            pass


def _flags():
    from gi.repository import Gio

    return Gio.DBusCallFlags.NONE


def _sig_flags():
    from gi.repository import Gio

    return Gio.DBusSignalFlags.NONE
