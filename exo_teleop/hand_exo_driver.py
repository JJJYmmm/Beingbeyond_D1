from __future__ import annotations

import re as _re
import threading as _th
import time as _tm
from dataclasses import dataclass as _dc
from typing import Dict as _D, List as _L, Optional as _O

import numpy as _np
import serial as _sr


@_dc
class HandExoCfg:
    p: str = "/dev/ttyCH343USB1"
    br: int = 115200
    to: float = 0.01
    dbg: bool = False

    ac: bool = True
    mx: int = 200
    mn: int = 30
    sw: int = 20
    se: float = 0.5

    off: _O[_L[float]] = None


class HandExoDriver:
    _RX = _re.compile(r"=\s*([-+]?\d+(?:\.\d+)?)")
    _N = 6

    def __init__(self, c: HandExoCfg):
        self.c = c
        if self.c.off is None:
            self.c.off = [40.0, -70.0, 60.0, 60.0, 60.0, 60.0]

        self._s: _O[_sr.Serial] = None
        self._st = _th.Event()
        self._thd: _O[_th.Thread] = None
        self._lk = _th.Lock()

        self._raw = _np.zeros(self._N, dtype=_np.float64)
        self._norm = _np.zeros(self._N, dtype=_np.float64)
        self._zero = _np.zeros(self._N, dtype=_np.float64)
        self._one = _np.asarray(self.c.off, dtype=_np.float64).copy()
        self._line = ""
        self._ts = 0.0
        self._nfrm = 0
        self._ready = False

        self._open()

        if self.c.ac:
            print("[hand_exo] auto calibration enabled")
            if not self.recalibrate():
                raise RuntimeError("[hand_exo] calib fail")
        else:
            print("[hand_exo] auto calibration disabled")

        self.start()

    def _open(self) -> None:
        try:
            self._s = _sr.Serial(port=self.c.p, baudrate=self.c.br, timeout=self.c.to)
            print(f"[hand_exo] open: {self.c.p} @ {self.c.br}")
        except _sr.SerialException as e:
            raise RuntimeError(f"[hand_exo] open fail: {e}") from e

    def start(self) -> None:
        if self._thd is not None and self._thd.is_alive():
            return
        self._st.clear()
        self._thd = _th.Thread(target=self._loop, name="hand_exo_rx", daemon=True)
        self._thd.start()

    def stop(self) -> None:
        self._st.set()
        if self._thd is not None:
            self._thd.join(timeout=1.0)
        if self._s is not None:
            try:
                self._s.close()
            except Exception:
                pass
        print("[hand_exo] closed")

    def ready(self) -> bool:
        with self._lk:
            return bool(self._ready)

    def wait_ready(self, timeout: float = 2.0) -> bool:
        print(f"[hand_exo] waiting ready... timeout={timeout:.2f}s")
        t0 = _tm.time()
        while _tm.time() - t0 < timeout:
            if self.ready():
                print("[hand_exo] ready")
                return True
            _tm.sleep(0.005)
        print("[hand_exo] ready timeout")
        return False

    def get_raw(self) -> _np.ndarray:
        with self._lk:
            return self._raw.copy()

    def get_norm(self) -> _np.ndarray:
        with self._lk:
            return self._norm.copy()

    def get_state(self) -> _D[str, _np.ndarray | float | int | str | bool]:
        with self._lk:
            return {
                "raw": self._raw.copy(),
                "norm": self._norm.copy(),
                "zero": self._zero.copy(),
                "one": self._one.copy(),
                "line": str(self._line),
                "ts": float(self._ts),
                "frames": int(self._nfrm),
                "ready": bool(self._ready),
            }

    def recalibrate(self) -> bool:
        if self._s is None:
            return False

        print("[hand_exo] calibration start: keep glove still at zero pose")

        ss: _L[_np.ndarray] = []
        ok = False

        while len(ss) < int(self.c.mx):
            try:
                b = self._s.readline()
            except _sr.SerialException:
                return False

            if not b:
                continue

            ln = b.decode("utf-8", errors="ignore").strip()
            a = self._parse(ln)
            if a is None:
                continue

            ss.append(a)

            if len(ss) >= int(self.c.sw):
                w = _np.stack(ss[-int(self.c.sw):], axis=0)
                rg = w.max(axis=0) - w.min(axis=0)
                if self.c.dbg:
                    print("[hand_exo] calib rg:", rg)
                if float(rg.max()) < float(self.c.se) and len(ss) >= int(self.c.mn):
                    ok = True
                    break

        if not ss or not ok:
            print("[hand_exo] calibration failed")
            return False

        use = _np.stack(ss[-int(self.c.sw):], axis=0)
        z = use.mean(axis=0)
        o = z + _np.asarray(self.c.off, dtype=_np.float64)

        with self._lk:
            self._zero[:] = z
            self._one[:] = o
            self._ready = True

        print(
            "[hand_exo] calibration ok: zero=",
            _np.array2string(z, precision=2, suppress_small=True),
            " one=",
            _np.array2string(o, precision=2, suppress_small=True),
            sep="",
        )
        return True

    def _parse(self, ln: str) -> _O[_np.ndarray]:
        if "ANGLES:" not in ln:
            return None
        m = self._RX.findall(ln)
        if len(m) != self._N:
            return None
        try:
            return _np.asarray([float(x) for x in m], dtype=_np.float64)
        except ValueError:
            return None

    def _map(self, a: _np.ndarray) -> _np.ndarray:
        with self._lk:
            z = self._zero.copy()
            o = self._one.copy()

        d = o - z
        d[_np.abs(d) < 1e-6] = 1.0
        t = (a - z) / d
        return _np.clip(t, 0.0, 1.0)

    def _loop(self) -> None:
        assert self._s is not None

        while not self._st.is_set():
            try:
                b = self._s.readline()
            except _sr.SerialException:
                break

            if not b:
                continue

            ln = b.decode("utf-8", errors="ignore").strip()
            a = self._parse(ln)
            if a is None:
                continue

            n = self._map(a)

            with self._lk:
                self._raw[:] = a
                self._norm[:] = n
                self._line = ln
                self._ts = _tm.time()
                self._nfrm += 1
                self._ready = True

            if self.c.dbg:
                print("[hand_exo]", _np.array2string(n, precision=3, suppress_small=True))