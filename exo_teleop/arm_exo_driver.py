from __future__ import annotations

import threading as _th
import time as _tm
from dataclasses import dataclass as _dc
from typing import Dict as _D, List as _L, Optional as _O, Sequence as _S

import numpy as _np
import serial as _sr


@_dc
class ArmExoCfg:
    p: str = "/dev/ttyCH343USB0"
    br: int = 1152000
    to: float = 0.01
    dbg: bool = False


class ArmExoDriver:
    _H = 0xAA
    _T = 0x55
    _FRAME_DEV_N = 7
    _N = 6
    _SZ = 17
    _RAW_MAX = 16384.0
    _DEG_MAX = 360.0

    def __init__(self, c: ArmExoCfg):
        self.c = c

        self._s: _O[_sr.Serial] = None
        self._st = _th.Event()
        self._thd: _O[_th.Thread] = None
        self._lk = _th.Lock()

        self._buf = bytearray()

        self._nfrm = 0
        self._nerr = 0
        self._ts = 0.0

        self._raw = _np.zeros(self._N, dtype=_np.int32)
        self._mod_deg = _np.zeros(self._N, dtype=_np.float64)
        self._abs_deg = _np.zeros(self._N, dtype=_np.float64)
        self._rel_deg = _np.zeros(self._N, dtype=_np.float64)
        self._valid = _np.zeros(self._N, dtype=_np.bool_)
        self._mask = 0

        self._last_mod: _L[_O[float]] = [None] * self._N
        self._zero_abs: _L[_O[float]] = [None] * self._N

        self._open()
        self.start()

    def _open(self) -> None:
        try:
            self._s = _sr.Serial(
                port=self.c.p,
                baudrate=self.c.br,
                timeout=self.c.to,
                bytesize=_sr.EIGHTBITS,
                parity=_sr.PARITY_NONE,
                stopbits=_sr.STOPBITS_ONE,
            )
            print(f"[arm_exo] open: {self.c.p} @ {self.c.br}")
        except _sr.SerialException as e:
            raise RuntimeError(f"[arm_exo] open fail: {e}") from e

    def start(self) -> None:
        if self._thd is not None and self._thd.is_alive():
            return
        self._st.clear()
        self._thd = _th.Thread(target=self._loop, name="arm_exo_rx", daemon=True)
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
        print("[arm_exo] closed")

    def zero(self, idx: _O[_S[int]] = None) -> None:
        with self._lk:
            if idx is None:
                idx = range(self._N)
            idx = list(idx)
            for i in idx:
                self._zero_abs[i] = float(self._abs_deg[i])

            z = [self._zero_abs[i] for i in range(self._N)]
        print(
            "[arm_exo] zero set:",
            _np.array2string(_np.asarray(z, dtype=_np.float64), precision=2, suppress_small=True),
        )

    def ready(self, min_valid: int = 1) -> bool:
        with self._lk:
            return int(self._valid.sum()) >= int(min_valid)

    def wait_ready(self, timeout: float = 2.0, min_valid: int = 1) -> bool:
        print(f"[arm_exo] waiting ready... min_valid={min_valid}, timeout={timeout:.2f}s")
        t0 = _tm.time()
        while _tm.time() - t0 < timeout:
            if self.ready(min_valid=min_valid):
                print("[arm_exo] ready")
                return True
            _tm.sleep(0.005)
        print("[arm_exo] ready timeout")
        return False

    def get_raw(self) -> _np.ndarray:
        with self._lk:
            return self._raw.copy()

    def get_deg(self) -> _np.ndarray:
        with self._lk:
            return self._rel_deg.copy()

    def get_rad(self) -> _np.ndarray:
        with self._lk:
            return _np.deg2rad(self._rel_deg.copy())

    def get_valid(self) -> _np.ndarray:
        with self._lk:
            return self._valid.copy()

    def get_state(self) -> _D[str, _np.ndarray | int | float]:
        with self._lk:
            return {
                "raw": self._raw.copy(),
                "deg_mod": self._mod_deg.copy(),
                "deg": self._rel_deg.copy(),
                "rad": _np.deg2rad(self._rel_deg.copy()),
                "valid": self._valid.copy(),
                "mask": int(self._mask),
                "ts": float(self._ts),
                "frames": int(self._nfrm),
                "errors": int(self._nerr),
            }

    def _find_h(self, b: bytearray) -> int:
        try:
            return b.index(self._H)
        except ValueError:
            return -1

    def _parse(self, fr: bytes) -> _O[tuple[int, _np.ndarray, _np.ndarray, _np.ndarray]]:
        if len(fr) != self._SZ:
            return None
        if fr[0] != self._H or fr[-1] != self._T:
            return None

        mask = int(fr[1])

        raw7 = _np.zeros(self._FRAME_DEV_N, dtype=_np.int32)
        deg7 = _np.zeros(self._FRAME_DEV_N, dtype=_np.float64)
        val7 = _np.zeros(self._FRAME_DEV_N, dtype=_np.bool_)

        for i in range(self._FRAME_DEV_N):
            hi = fr[2 + 2 * i]
            lo = fr[2 + 2 * i + 1]
            x = (hi << 8) | lo
            raw7[i] = x
            deg7[i] = (float(x) / self._RAW_MAX) * self._DEG_MAX
            val7[i] = bool((mask >> i) & 0x01)

        raw = raw7[: self._N].copy()
        deg = deg7[: self._N].copy()
        val = val7[: self._N].copy()
        mask6 = mask & 0x3F
        return mask6, raw, deg, val

    def _upd(self, mask: int, raw: _np.ndarray, deg: _np.ndarray, val: _np.ndarray) -> None:
        now = _tm.time()

        with self._lk:
            self._mask = mask
            self._raw[:] = raw
            self._mod_deg[:] = deg
            self._valid[:] = val
            self._ts = now
            self._nfrm += 1

            for i in range(self._N):
                if not bool(val[i]):
                    continue

                cur = float(deg[i])
                prev = self._last_mod[i]

                if prev is None:
                    self._abs_deg[i] = cur
                    self._last_mod[i] = cur
                    if self._zero_abs[i] is None:
                        self._zero_abs[i] = self._abs_deg[i]
                else:
                    d = cur - prev
                    if d > 180.0:
                        d -= 360.0
                    elif d < -180.0:
                        d += 360.0
                    self._abs_deg[i] += d
                    self._last_mod[i] = cur

                z = self._zero_abs[i]
                if z is None:
                    z = self._abs_deg[i]
                    self._zero_abs[i] = z
                self._rel_deg[i] = self._abs_deg[i] - z

    def _loop(self) -> None:
        assert self._s is not None

        while not self._st.is_set():
            try:
                n = self._s.in_waiting
                b = self._s.read(n if n > 0 else 1)
            except _sr.SerialException:
                with self._lk:
                    self._nerr += 1
                break

            if not b:
                continue

            self._buf.extend(b)

            while len(self._buf) >= self._SZ:
                p = self._find_h(self._buf)
                if p < 0:
                    if len(self._buf) > self._SZ - 1:
                        self._buf = self._buf[-(self._SZ - 1):]
                    break

                if p > 0:
                    self._buf = self._buf[p:]

                if len(self._buf) < self._SZ:
                    break

                fr = bytes(self._buf[: self._SZ])
                del self._buf[: self._SZ]

                out = self._parse(fr)
                if out is None:
                    with self._lk:
                        self._nerr += 1
                    continue

                self._upd(*out)

                if self.c.dbg:
                    s = self.get_state()
                    print(
                        "[arm_exo]",
                        "mask=0x%02X" % s["mask"],
                        "deg=",
                        _np.array2string(s["deg"], precision=2, suppress_small=True),
                        "valid=",
                        s["valid"].astype(_np.int32),
                    )