"""Store seam tests — the same suite runs against both backends:

- ``py``:   PyStore, pure Python over sparse files.
- ``rust``: kiokud daemon (compiled once per session), via Unix socket.

The two must be interchangeable down to the on-disk blob format.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from engine.store import (
    DISK_PLANET_BYTES,
    DISK_VIRTUAL_BYTES,
    PLANET_CELL_MASK,
    VRAM_VIRTUAL_BYTES,
    BlobHandle,
    Cell,
    KiokudStore,
    PyStore,
    StoreError,
    build_kiokud,
    hash64,
    keyword_cell,
    spawn_kiokud,
)

CEILING = 256 << 20  # deterministic small ceiling for both backends
BUDGET = 8 << 20

_OBJECT_HEADER_BYTES = 16


@pytest.fixture(scope="session")
def kiokud_binary(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    if shutil.which("rustc") is None:
        return None
    substrate = Path(__file__).resolve().parents[2] / "substrate"
    out = tmp_path_factory.mktemp("kiokud") / "kiokud"
    binary = build_kiokud(substrate, out)
    assert binary is not None, "rustc is present but kiokud failed to build"
    return binary


@pytest.fixture(params=["py", "rust"])
def backend(request: pytest.FixtureRequest, tmp_path: Path, kiokud_binary: Path | None):
    """Yields (store, disk_path). disk_path is the 4 TiB sparse blob file —
    identical format on both backends, used by the corruption test."""
    if request.param == "py":
        store = PyStore(tmp_path / "pystore", ceiling_bytes=CEILING)
        yield store, store.disk_path
        store.close()
    else:
        if kiokud_binary is None:
            pytest.skip("rustc not available")
        socket_path = f"/tmp/kiokud_test_{os.getpid()}_{uuid.uuid4().hex[:8]}.sock"
        disk_path = tmp_path / "kioku_box.disk"
        proc = spawn_kiokud(kiokud_binary, socket_path, disk_path, ceiling_bytes=CEILING)
        store = KiokudStore(socket_path)
        yield store, disk_path
        store.close()
        proc.terminate()
        proc.wait(timeout=5)
        if os.path.exists(socket_path):
            os.unlink(socket_path)


def test_open_put_get_roundtrip(backend) -> None:
    store, _ = backend
    space = store.open_space(BUDGET)
    cells = [Cell(cell=4242, act=0.75, expert=3, weight=999), Cell(cell=1, act=-1.5, expert=0, weight=1)]
    assert store.put_cells(space, cells) == 2
    got = store.get_cell(space, 4242)
    assert got == Cell(cell=4242, act=0.75, expert=3, weight=999)
    assert store.get_cell(space, 1).act == -1.5


def test_untouched_cell_is_none(backend) -> None:
    store, _ = backend
    space = store.open_space(BUDGET)
    assert store.get_cell(space, 123456) is None


def test_closed_space_is_refused(backend) -> None:
    store, _ = backend
    with pytest.raises(StoreError):
        store.get_cell(9999, 0)


def test_isolation_between_spaces(backend) -> None:
    store, _ = backend
    s1 = store.open_space(BUDGET)
    s2 = store.open_space(BUDGET)
    store.put_cells(s1, [Cell(cell=7, act=1.5, expert=1, weight=11)])
    store.put_cells(s2, [Cell(cell=7, act=9.5, expert=2, weight=22)])
    assert store.get_cell(s1, 7).act == 1.5
    assert store.get_cell(s2, 7).act == 9.5
    # A segment s2 never touched stays uncommitted.
    assert store.get_cell(s2, 1 << 21) is None


def test_address_masking_wraps(backend) -> None:
    store, _ = backend
    space = store.open_space(BUDGET)
    beyond = (5 << 30) | 4242  # high bits outside the planet
    store.put_cells(space, [Cell(cell=beyond, act=2.0, expert=0, weight=5)])
    got = store.get_cell(space, beyond & PLANET_CELL_MASK)
    assert got is not None and got.weight == 5


def test_scan_returns_only_written_cells(backend) -> None:
    store, _ = backend
    space = store.open_space(BUDGET)
    store.put_cells(space, [Cell(cell=c, act=1.0, expert=0, weight=c) for c in (100, 105, 199)])
    found = store.scan(space, 100, 100)
    assert [c.cell for c in found] == [100, 105, 199]
    assert all(c.weight == c.cell for c in found)


def test_blob_roundtrip(backend) -> None:
    store, _ = backend
    space = store.open_space(BUDGET)
    payload = bytes(i % 251 for i in range(100_000))
    handle = store.put_blob(space, payload)
    assert handle.length == len(payload)
    assert store.get_blob(space, handle) == payload
    # Sequential blobs land on later blocks.
    h2 = store.put_blob(space, b"second engram")
    assert h2.block > handle.block
    assert store.get_blob(space, h2) == b"second engram"


def test_blob_cross_space_is_refused(backend) -> None:
    store, _ = backend
    s1 = store.open_space(BUDGET)
    s2 = store.open_space(BUDGET)
    handle = store.put_blob(s1, b"private to s1")
    with pytest.raises(StoreError):
        store.get_blob(s2, handle)


def test_blob_corruption_is_detected(backend) -> None:
    """Flip one payload byte directly in the sparse disk file — both
    backends must refuse on CRC. Also proves the formats are identical."""
    store, disk_path = backend
    space = store.open_space(BUDGET)
    handle = store.put_blob(space, b"important findings")
    offset = space * DISK_PLANET_BYTES + handle.block * 4096 + _OBJECT_HEADER_BYTES + 2
    with open(disk_path, "r+b") as f:
        f.seek(offset)
        byte = f.read(1)
        f.seek(offset)
        f.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(StoreError):
        store.get_blob(space, handle)


def test_budget_check(backend) -> None:
    store, _ = backend
    space = store.open_space(300 * 1024)  # one segment + table fits, two don't
    store.put_cells(space, [Cell(cell=0, act=1.0, expert=0, weight=1)])
    report = store.check_budget(space)
    assert report.within and report.committed <= report.budget
    # Touch a second segment: 2 * 256 KiB > 300 KiB.
    store.put_cells(space, [Cell(cell=1 << 20, act=1.0, expert=0, weight=1)])
    report = store.check_budget(space)
    assert not report.within


def test_ceiling_refuses_not_degrades(backend) -> None:
    store, _ = backend
    with pytest.raises(StoreError):
        store.open_space(CEILING + 1)


def test_stats_gauges(backend) -> None:
    store, _ = backend
    space = store.open_space(BUDGET)
    before = store.stats()
    assert before.vram_virtual == VRAM_VIRTUAL_BYTES == 1 << 40
    assert before.disk_virtual == DISK_VIRTUAL_BYTES == 1 << 42
    assert before.ceiling == CEILING
    assert before.open_spaces == 1
    store.put_cells(space, [Cell(cell=0, act=1.0, expert=0, weight=1)])
    after = store.stats()
    assert after.vram_committed > before.vram_committed
    assert after.spaces[0].space == space
    assert after.spaces[0].committed > 0
    # Small outside, huge inside.
    assert after.vram_committed < after.vram_virtual // 1000


def test_release_space_resets(backend) -> None:
    store, _ = backend
    space = store.open_space(BUDGET)
    store.put_cells(space, [Cell(cell=0, act=1.0, expert=0, weight=1)])
    handle = store.put_blob(space, b"to be forgotten")
    freed = store.release_space(space)
    assert freed > 0
    with pytest.raises(StoreError):
        store.get_cell(space, 0)
    # The planet is reused, and comes back clean.
    again = store.open_space(BUDGET)
    assert again == space
    assert store.get_cell(again, 0) is None
    with pytest.raises(StoreError):
        store.get_blob(again, handle)
    h2 = store.put_blob(again, b"fresh start")
    assert h2.block == 1  # first data block again


def test_pystore_persists_across_reopen(tmp_path: Path) -> None:
    store = PyStore(tmp_path / "pystore", ceiling_bytes=CEILING)
    space = store.open_space(BUDGET)
    store.put_cells(space, [Cell(cell=42, act=3.25, expert=7, weight=1234)])
    handle = store.put_blob(space, b"survives a restart")
    store.close()

    reopened = PyStore(tmp_path / "pystore", ceiling_bytes=CEILING)
    assert reopened.get_cell(space, 42) == Cell(cell=42, act=3.25, expert=7, weight=1234)
    assert reopened.get_blob(space, handle) == b"survives a restart"
    # Cursor restored from the superblock: next blob lands after the first.
    h2 = reopened.put_blob(space, b"next")
    assert h2.block > handle.block
    reopened.close()


def test_hash64_known_fnv1a_vectors() -> None:
    assert hash64("") == 0xCBF29CE484222325
    assert hash64("a") == 0xAF63DC4C8601EC8C
    assert hash64("foobar") == 0x85944171F73967E8


def test_keyword_cell_is_deterministic_and_in_range() -> None:
    for word in ("hanami", "qwen", "記憶", "x" * 500):
        cell = keyword_cell(word)
        assert 0 <= cell <= PLANET_CELL_MASK
        assert cell == keyword_cell(word)


def test_blob_handle_is_plain_data() -> None:
    h = BlobHandle(block=5, length=100)
    assert (h.block, h.length) == (5, 100)
