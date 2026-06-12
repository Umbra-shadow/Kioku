"""Kioku store — the seam between the engine and the Cadran substrate.

Two interchangeable backends behind one protocol:

- :class:`KiokudStore` — speaks newline-delimited JSON to ``kiokud`` (the
  Rust daemon that owns the 1 TiB sparse vRAM universe and the 4 TiB
  virtual disk) over a Unix socket. This is the headline path.
- :class:`PyStore` — pure-Python fallback used when ``rustc`` (or the
  daemon) is unavailable. Same planet/segment/mask arithmetic, same
  on-disk blob format (superblocks, CRC-verified objects), over sparse
  files. Interchangeable down to the byte layout of the virtual disk.

Addressing discipline (documented in docs/MEMORY_MODEL.md): the keyword
index lives at deterministic cell addresses — ``hash64(keyword) &
PLANET_CELL_MASK`` — so a lookup is one shift+mask jump, never a search.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
import zlib
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

log = logging.getLogger("kioku.store")

# --- Substrate geometry (mirrors substrate/cadran_vram.rs exactly) ----------

PLANET_BITS = 14
PLANET_CELL_BITS = 22
SEGMENT_CELL_BITS = 14

NUM_PLANETS = 1 << PLANET_BITS
CELLS_PER_PLANET = 1 << PLANET_CELL_BITS
CELLS_PER_SEGMENT = 1 << SEGMENT_CELL_BITS
SEGMENTS_PER_PLANET = CELLS_PER_PLANET // CELLS_PER_SEGMENT

PLANET_CELL_MASK = CELLS_PER_PLANET - 1
SEGMENT_CELL_MASK = CELLS_PER_SEGMENT - 1

CELL_BYTES = 16  # struct VRAMCell { f32 act, u32 expert, u64 weight }
_CELL_STRUCT = struct.Struct("<fIQ")
SEGMENT_BYTES = CELLS_PER_SEGMENT * CELL_BYTES
PLANET_TABLE_BYTES = 2048  # SEGMENTS_PER_PLANET pointers, 8 B each
VRAM_VIRTUAL_BYTES = NUM_PLANETS * CELLS_PER_PLANET * CELL_BYTES  # 1 TiB

# --- Disk geometry (mirrors substrate/cadran_storage.rs exactly) ------------

DISK_BLOCK_BYTES = 4096
DISK_PLANET_BLOCK_BITS = 16
DISK_BLOCKS_PER_PLANET = 1 << DISK_PLANET_BLOCK_BITS
DISK_PLANET_BYTES = DISK_BLOCKS_PER_PLANET * DISK_BLOCK_BYTES  # 256 MiB
DISK_VIRTUAL_BYTES = NUM_PLANETS * DISK_PLANET_BYTES  # 4 TiB

_SUPERBLOCK_MAGIC = 0x4341_4452_4449_534B  # "CADRDISK"
_SUPERBLOCK_VERSION = 2
_OBJECT_MAGIC = 0x4F42_4A31  # "OBJ1"
_OBJECT_HEADER = struct.Struct("<IIQ")  # magic, crc32, payload len
_FIRST_DATA_BLOCK = 1

DEFAULT_SOCKET = "/tmp/kiokud.sock"

# --- Addressing --------------------------------------------------------------

_FNV64_OFFSET = 0xCBF2_9CE4_8422_2325
_FNV64_PRIME = 0x0000_0100_0000_01B3
_U64 = (1 << 64) - 1


def hash64(text: str) -> int:
    """FNV-1a 64-bit. The one hash both sides of the seam agree on."""
    h = _FNV64_OFFSET
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * _FNV64_PRIME) & _U64
    return h


def keyword_cell(keyword: str) -> int:
    """Deterministic index cell for a keyword: one shift+mask jump."""
    return hash64(keyword) & PLANET_CELL_MASK


# --- Protocol types -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cell:
    cell: int
    act: float
    expert: int
    weight: int


@dataclass(frozen=True, slots=True)
class BlobHandle:
    block: int
    length: int


@dataclass(frozen=True, slots=True)
class BudgetReport:
    within: bool
    committed: int
    budget: int


@dataclass(frozen=True, slots=True)
class SpaceInfo:
    space: int
    budget: int
    committed: int


@dataclass(frozen=True, slots=True)
class StoreStats:
    backend: str
    vram_committed: int
    vram_virtual: int
    disk_committed: int
    disk_virtual: int
    reserved: int
    ceiling: int
    open_spaces: int
    spaces: tuple[SpaceInfo, ...] = field(default_factory=tuple)


class StoreError(RuntimeError):
    """Any refusal or failure at the substrate seam."""


class MemoryStore(Protocol):
    """What the engine needs from a substrate, regardless of backend."""

    def open_space(self, budget: int) -> int: ...

    def put_cells(self, space: int, cells: Iterable[Cell]) -> int: ...

    def get_cell(self, space: int, cell: int) -> Cell | None: ...

    def scan(self, space: int, start: int, count: int) -> list[Cell]: ...

    def put_blob(self, space: int, payload: bytes) -> BlobHandle: ...

    def get_blob(self, space: int, handle: BlobHandle) -> bytes: ...

    def check_budget(self, space: int) -> BudgetReport: ...

    def stats(self) -> StoreStats: ...

    def release_space(self, space: int) -> int: ...

    def close(self) -> None: ...


def _stats_from_dict(backend: str, d: dict) -> StoreStats:
    return StoreStats(
        backend=backend,
        vram_committed=int(d["vram_committed"]),
        vram_virtual=int(d["vram_virtual"]),
        disk_committed=int(d["disk_committed"]),
        disk_virtual=int(d["disk_virtual"]),
        reserved=int(d["reserved"]),
        ceiling=int(d["ceiling"]),
        open_spaces=int(d["open_spaces"]),
        spaces=tuple(
            SpaceInfo(int(s["space"]), int(s["budget"]), int(s["committed"]))
            for s in d.get("spaces", [])
        ),
    )


# --- KiokudStore: the Rust path ----------------------------------------------


class KiokudStore:
    """Client for the kiokud daemon: one JSON object per line, both ways."""

    def __init__(self, socket_path: str = DEFAULT_SOCKET, timeout: float = 10.0) -> None:
        self._socket_path = socket_path
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        try:
            self._sock.connect(socket_path)
        except OSError as e:
            self._sock.close()
            raise StoreError(f"cannot reach kiokud at {socket_path}: {e}") from e
        self._file = self._sock.makefile("rwb")

    def _request(self, payload: dict) -> dict:
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        with self._lock:
            try:
                self._file.write(line.encode("utf-8"))
                self._file.flush()
                raw = self._file.readline()
            except OSError as e:
                raise StoreError(f"kiokud connection failed: {e}") from e
        if not raw:
            raise StoreError("kiokud closed the connection")
        response = json.loads(raw)
        if not response.get("ok"):
            raise StoreError(response.get("error", "unknown kiokud error"))
        return response

    def ping(self) -> bool:
        return bool(self._request({"op": "ping"}).get("pong"))

    def open_space(self, budget: int) -> int:
        return int(self._request({"op": "open_space", "budget": budget})["space"])

    def put_cells(self, space: int, cells: Iterable[Cell]) -> int:
        batch = [
            {"cell": c.cell, "act": c.act, "expert": c.expert, "weight": c.weight}
            for c in cells
        ]
        if not batch:
            return 0
        r = self._request({"op": "put", "space": space, "cells": batch})
        return int(r["written"])

    def get_cell(self, space: int, cell: int) -> Cell | None:
        r = self._request({"op": "get", "space": space, "cell": cell})
        if not r.get("found"):
            return None
        return Cell(cell=cell, act=float(r["act"]), expert=int(r["expert"]), weight=int(r["weight"]))

    def scan(self, space: int, start: int, count: int) -> list[Cell]:
        r = self._request({"op": "scan", "space": space, "start": start, "count": count})
        return [
            Cell(cell=int(c["cell"]), act=float(c["act"]), expert=int(c["expert"]), weight=int(c["weight"]))
            for c in r["cells"]
        ]

    def put_blob(self, space: int, payload: bytes) -> BlobHandle:
        b64 = b64encode(payload).decode("ascii")
        r = self._request({"op": "put_blob", "space": space, "b64": b64})
        return BlobHandle(block=int(r["block"]), length=int(r["len"]))

    def get_blob(self, space: int, handle: BlobHandle) -> bytes:
        r = self._request(
            {"op": "get_blob", "space": space, "block": handle.block, "len": handle.length}
        )
        return b64decode(r["b64"])

    def check_budget(self, space: int) -> BudgetReport:
        r = self._request({"op": "check_budget", "space": space})
        return BudgetReport(bool(r["within"]), int(r["committed"]), int(r["budget"]))

    def stats(self) -> StoreStats:
        return _stats_from_dict("kiokud", self._request({"op": "stats"}))

    def release_space(self, space: int) -> int:
        return int(self._request({"op": "release_space", "space": space})["freed"])

    def close(self) -> None:
        with self._lock:
            try:
                self._file.close()
            finally:
                self._sock.close()


# --- PyStore: the pure-Python fallback ---------------------------------------


def _host_ram_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 1 << 30


def _punch_hole(fd: int, offset: int, length: int) -> bool:
    """Linux fallocate(FALLOC_FL_PUNCH_HOLE|KEEP_SIZE); False if unsupported."""
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        r = libc.fallocate(
            ctypes.c_int(fd),
            ctypes.c_int(0x01 | 0x02),  # KEEP_SIZE | PUNCH_HOLE
            ctypes.c_int64(offset),
            ctypes.c_int64(length),
        )
        return r == 0
    except (OSError, AttributeError):
        return False


class PyStore:
    """Pure-Python substrate with the same arithmetic as the Rust path.

    Layout under ``data_dir``:
      - ``vram.bin``     1 TiB sparse file of 16-byte cells (pread/pwrite)
      - ``touched.bin``  one bit per (planet, segment): the commit map
      - ``disk.bin``     4 TiB sparse file, byte-compatible with cadran_storage
      - ``meta.json``    open spaces and their budgets (persists across runs)
    """

    def __init__(self, data_dir: str | Path, ceiling_bytes: int | None = None) -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ceiling = ceiling_bytes if ceiling_bytes is not None else _host_ram_bytes() // 2

        self._vram_fd = self._open_sparse(self._dir / "vram.bin", VRAM_VIRTUAL_BYTES)
        self._disk_fd = self._open_sparse(self._dir / "disk.bin", DISK_VIRTUAL_BYTES)

        bitmap_bytes = NUM_PLANETS * SEGMENTS_PER_PLANET // 8
        self._bitmap_path = self._dir / "touched.bin"
        if self._bitmap_path.exists():
            self._bitmap = bytearray(self._bitmap_path.read_bytes())
            if len(self._bitmap) != bitmap_bytes:
                self._bitmap = bytearray(bitmap_bytes)
        else:
            self._bitmap = bytearray(bitmap_bytes)

        self._meta_path = self._dir / "meta.json"
        self._spaces: dict[int, int] = {}  # planet -> budget
        if self._meta_path.exists():
            raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._spaces = {int(k): int(v) for k, v in raw.get("spaces", {}).items()}
        self._cursors: dict[int, int] = {}  # planet -> next free block

        log.warning(
            "PyStore active (pure-Python fallback over sparse files at %s) — "
            "the Rust kiokud path is the headline; this keeps the demo alive without rustc",
            self._dir,
        )

    @property
    def disk_path(self) -> Path:
        return self._dir / "disk.bin"

    @staticmethod
    def _open_sparse(path: Path, size: int) -> int:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        if os.fstat(fd).st_size != size:
            os.ftruncate(fd, size)
        return fd

    # -- internal helpers ------------------------------------------------

    def _require_open(self, space: int) -> None:
        if space not in self._spaces:
            raise StoreError(f"space SpaceId({space}) is closed")

    def _seg_index(self, planet: int, local_cell: int) -> int:
        return planet * SEGMENTS_PER_PLANET + ((local_cell & PLANET_CELL_MASK) >> SEGMENT_CELL_BITS)

    def _touched(self, planet: int, local_cell: int) -> bool:
        i = self._seg_index(planet, local_cell)
        return bool(self._bitmap[i >> 3] & (1 << (i & 7)))

    def _touch(self, planet: int, local_cell: int) -> None:
        i = self._seg_index(planet, local_cell)
        self._bitmap[i >> 3] |= 1 << (i & 7)

    def _flush_bitmap(self) -> None:
        self._bitmap_path.write_bytes(bytes(self._bitmap))

    def _save_meta(self) -> None:
        self._meta_path.write_text(
            json.dumps({"spaces": {str(k): v for k, v in self._spaces.items()}}),
            encoding="utf-8",
        )

    def _cell_offset(self, planet: int, local_cell: int) -> int:
        return ((planet << PLANET_CELL_BITS) | (local_cell & PLANET_CELL_MASK)) * CELL_BYTES

    def _planet_segments(self, planet: int) -> int:
        base = planet * SEGMENTS_PER_PLANET
        return sum(
            1
            for i in range(base, base + SEGMENTS_PER_PLANET)
            if self._bitmap[i >> 3] & (1 << (i & 7))
        )

    def _planet_committed(self, planet: int) -> int:
        segs = self._planet_segments(planet)
        return segs * SEGMENT_BYTES + (PLANET_TABLE_BYTES if segs else 0)

    # -- disk (blob) helpers: byte-compatible with cadran_storage.rs ------

    def _disk_base(self, planet: int) -> int:
        return (planet & (NUM_PLANETS - 1)) * DISK_PLANET_BYTES

    def _cursor(self, planet: int) -> int:
        if planet in self._cursors:
            return self._cursors[planet]
        raw = os.pread(self._disk_fd, 24, self._disk_base(planet))
        magic = int.from_bytes(raw[0:8], "little")
        if magic == _SUPERBLOCK_MAGIC:
            cursor = int.from_bytes(raw[16:24], "little")
            if not _FIRST_DATA_BLOCK <= cursor <= DISK_BLOCKS_PER_PLANET:
                raise StoreError("corrupt superblock cursor")
        else:
            cursor = _FIRST_DATA_BLOCK
            self._write_superblock(planet, cursor)
        self._cursors[planet] = cursor
        return cursor

    def _write_superblock(self, planet: int, cursor: int) -> None:
        b = (
            _SUPERBLOCK_MAGIC.to_bytes(8, "little")
            + _SUPERBLOCK_VERSION.to_bytes(4, "little")
            + b"\x00" * 4
            + cursor.to_bytes(8, "little")
        )
        os.pwrite(self._disk_fd, b, self._disk_base(planet))

    # -- MemoryStore interface --------------------------------------------

    def open_space(self, budget: int) -> int:
        with self._lock:
            reserved = sum(self._spaces.values())
            if reserved + budget > self._ceiling:
                raise StoreError(
                    f"host-safe ceiling: requested {budget} B would push reservations "
                    f"past {reserved} B (ceiling {self._ceiling} B)"
                )
            planet = next(p for p in range(1, NUM_PLANETS) if p not in self._spaces)
            self._spaces[planet] = budget
            self._save_meta()
            return planet

    def put_cells(self, space: int, cells: Iterable[Cell]) -> int:
        with self._lock:
            self._require_open(space)
            written = 0
            for c in cells:
                local = c.cell & PLANET_CELL_MASK
                os.pwrite(
                    self._vram_fd,
                    _CELL_STRUCT.pack(c.act, c.expert, c.weight),
                    self._cell_offset(space, local),
                )
                self._touch(space, local)
                written += 1
            if written:
                self._flush_bitmap()
            return written

    def get_cell(self, space: int, cell: int) -> Cell | None:
        with self._lock:
            self._require_open(space)
            local = cell & PLANET_CELL_MASK
            if not self._touched(space, local):
                return None
            raw = os.pread(self._vram_fd, CELL_BYTES, self._cell_offset(space, local))
            act, expert, weight = _CELL_STRUCT.unpack(raw)
            return Cell(cell=cell, act=act, expert=expert, weight=weight)

    def scan(self, space: int, start: int, count: int) -> list[Cell]:
        with self._lock:
            self._require_open(space)
            out: list[Cell] = []
            cell = start
            end = start + min(count, 1 << 16)
            while cell < end:
                local = cell & PLANET_CELL_MASK
                if not self._touched(space, local):
                    # Skip the rest of this untouched segment.
                    cell = (cell | SEGMENT_CELL_MASK) + 1
                    continue
                raw = os.pread(self._vram_fd, CELL_BYTES, self._cell_offset(space, local))
                act, expert, weight = _CELL_STRUCT.unpack(raw)
                if act != 0.0 or expert != 0 or weight != 0:
                    out.append(Cell(cell=cell, act=act, expert=expert, weight=weight))
                cell += 1
            return out

    def put_blob(self, space: int, payload: bytes) -> BlobHandle:
        with self._lock:
            self._require_open(space)
            cursor = self._cursor(space)
            total = _OBJECT_HEADER.size + len(payload)
            blocks_needed = -(-total // DISK_BLOCK_BYTES)
            if cursor + blocks_needed > DISK_BLOCKS_PER_PLANET:
                raise StoreError("planet disk full (256 MiB room)")
            offset = self._disk_base(space) + cursor * DISK_BLOCK_BYTES
            header = _OBJECT_HEADER.pack(_OBJECT_MAGIC, zlib.crc32(payload), len(payload))
            os.pwrite(self._disk_fd, header + payload, offset)
            self._write_superblock(space, cursor + blocks_needed)
            self._cursors[space] = cursor + blocks_needed
            return BlobHandle(block=cursor, length=len(payload))

    def get_blob(self, space: int, handle: BlobHandle) -> bytes:
        with self._lock:
            self._require_open(space)
            offset = self._disk_base(space) + (
                handle.block & (DISK_BLOCKS_PER_PLANET - 1)
            ) * DISK_BLOCK_BYTES
            header = os.pread(self._disk_fd, _OBJECT_HEADER.size, offset)
            magic, stored_crc, length = _OBJECT_HEADER.unpack(header)
            if magic != _OBJECT_MAGIC:
                raise StoreError("no object at handle")
            if length != handle.length or length > DISK_PLANET_BYTES:
                raise StoreError("object length mismatch")
            payload = os.pread(self._disk_fd, length, offset + _OBJECT_HEADER.size)
            if zlib.crc32(payload) != stored_crc:
                raise StoreError("object CRC mismatch")
            return payload

    def check_budget(self, space: int) -> BudgetReport:
        with self._lock:
            self._require_open(space)
            committed = self._planet_committed(space)
            budget = self._spaces[space]
            return BudgetReport(committed <= budget, committed, budget)

    def stats(self) -> StoreStats:
        with self._lock:
            total_segs = sum(bin(b).count("1") for b in self._bitmap)
            planets_touched = sum(1 for p in self._spaces if self._planet_segments(p))
            vram_committed = (
                total_segs * SEGMENT_BYTES
                + planets_touched * PLANET_TABLE_BYTES
                + NUM_PLANETS * 8
            )
            disk_committed = os.fstat(self._disk_fd).st_blocks * 512
            return StoreStats(
                backend="pystore",
                vram_committed=vram_committed,
                vram_virtual=VRAM_VIRTUAL_BYTES,
                disk_committed=disk_committed,
                disk_virtual=DISK_VIRTUAL_BYTES,
                reserved=sum(self._spaces.values()),
                ceiling=self._ceiling,
                open_spaces=len(self._spaces),
                spaces=tuple(
                    SpaceInfo(p, b, self._planet_committed(p))
                    for p, b in sorted(self._spaces.items())
                ),
            )

    def release_space(self, space: int) -> int:
        with self._lock:
            self._require_open(space)
            freed = self._planet_committed(space)
            vram_base = self._cell_offset(space, 0)
            planet_vram = CELLS_PER_PLANET * CELL_BYTES
            if not _punch_hole(self._vram_fd, vram_base, planet_vram):
                # No hole punching: zero only the touched segments.
                zeros = b"\x00" * SEGMENT_BYTES
                for seg in range(SEGMENTS_PER_PLANET):
                    if self._touched(space, seg << SEGMENT_CELL_BITS):
                        os.pwrite(self._vram_fd, zeros, vram_base + seg * SEGMENT_BYTES)
            base = space * SEGMENTS_PER_PLANET
            for i in range(base, base + SEGMENTS_PER_PLANET):
                self._bitmap[i >> 3] &= ~(1 << (i & 7)) & 0xFF
            self._flush_bitmap()
            _punch_hole(self._disk_fd, self._disk_base(space), DISK_PLANET_BYTES)
            self._write_superblock(space, _FIRST_DATA_BLOCK)
            self._cursors[space] = _FIRST_DATA_BLOCK
            del self._spaces[space]
            self._save_meta()
            return freed

    def close(self) -> None:
        with self._lock:
            self._flush_bitmap()
            os.close(self._vram_fd)
            os.close(self._disk_fd)


# --- Factory: prefer the Rust daemon, fall back to PyStore --------------------


def build_kiokud(substrate_dir: str | Path, out_path: str | Path) -> Path | None:
    """Compile kiokud with bare rustc. None if rustc is missing or fails."""
    if shutil.which("rustc") is None:
        return None
    src = Path(substrate_dir) / "kiokud.rs"
    out = Path(out_path)
    result = subprocess.run(
        ["rustc", "--edition=2021", "-C", "opt-level=3", str(src), "-o", str(out)],
        capture_output=True,
        text=True,
        cwd=substrate_dir,
    )
    if result.returncode != 0:
        log.error("kiokud build failed: %s", result.stderr[-2000:])
        return None
    return out


def spawn_kiokud(
    binary: str | Path,
    socket_path: str,
    disk_path: str | Path,
    ceiling_bytes: int | None = None,
    startup_timeout: float = 5.0,
) -> subprocess.Popen[bytes]:
    """Start the daemon and wait until its socket answers."""
    env = os.environ.copy()
    env["KIOKUD_SOCKET"] = socket_path
    env["KIOKUD_DISK"] = str(disk_path)
    if ceiling_bytes is not None:
        env["KIOKUD_CEILING_BYTES"] = str(ceiling_bytes)
    proc = subprocess.Popen([str(binary)], env=env)
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise StoreError(f"kiokud exited at startup (code {proc.returncode})")
        try:
            KiokudStore(socket_path).close()
            return proc
        except StoreError:
            time.sleep(0.05)
    proc.terminate()
    raise StoreError("kiokud did not come up in time")


def open_store(data_dir: str | Path, prefer: str | None = None) -> MemoryStore:
    """Open the best available substrate.

    ``prefer`` (or env ``KIOKU_STORE``): ``rust`` | ``py`` | ``auto`` (default).
    Order in auto mode: existing daemon socket -> build & spawn kiokud ->
    PyStore fallback (logged loudly, never a crash).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    prefer = (prefer or os.environ.get("KIOKU_STORE", "auto")).lower()
    socket_path = os.environ.get("KIOKU_SOCKET", DEFAULT_SOCKET)

    if prefer == "py":
        return PyStore(data_dir / "pystore")

    try:
        store = KiokudStore(socket_path)
        store.ping()
        log.info("connected to running kiokud at %s", socket_path)
        return store
    except StoreError:
        pass

    substrate_dir = Path(__file__).resolve().parent.parent / "substrate"
    binary = build_kiokud(substrate_dir, data_dir / "kiokud")
    if binary is not None:
        try:
            spawn_kiokud(binary, socket_path, data_dir / "kioku_box.disk")
            store = KiokudStore(socket_path)
            store.ping()
            log.info("spawned kiokud at %s", socket_path)
            return store
        except StoreError as e:
            log.error("kiokud spawn failed: %s", e)

    if prefer == "rust":
        raise StoreError("rust store requested but kiokud is unavailable")
    return PyStore(data_dir / "pystore")
