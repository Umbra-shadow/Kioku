# compute/v3 — Stage 1 complete: the Space bundle and the host-safe Box

v2 was the substrate (vRAM, vGPU, disk). v3 closes Stage 1 of the
roadmap: **`Space`** and the **host-safe ceiling**. The box is now a
governed building, not a pile of hardware. **23 tests, zero warnings,
stable Rust.**

## Files

| File | Role | vs v2 |
|------|------|-------|
| `cadran_vram.rs` | 1 TiB sparse universe | + `planet_committed_bytes` (per-space accounting) |
| `cadran_vgpu.rs` | Lane engine + kernels + demo `main` | + The Box demo section |
| `cadran_storage.rs` | 4 TiB virtual disk | raw block I/O past a planet boundary is now **refused** (roadmap acceptance), never redirected |
| `space.rs` | **new** — `Space`, `TheBox`, capabilities, ceiling | — |

## What changed

### `space.rs` — one space, one unit, one door
A space bundles **(vRAM planet, disk planet, capabilities)** under one
`SpaceId`. `TheBox` owns the universe and the disk; it is the only way
to open or release a space.

- `open_space(budget, capabilities)` — reserves the budget against the
  ceiling, assigns a free planet (released planets are reused — tested
  clean on reuse).
- `space(id)` → `SpaceHandle` — cell read/write/peek, `put_paper` /
  `get_paper` (a handle from another space is **refused**), per-space
  `committed_bytes`, `check_budget`.
- `release_space(id)` — vRAM planet reclaimed + disk hole punched +
  reservation returned, atomically from the caller's view.
- **Capabilities default off** (`network`, `host_filesystem`), granted
  only by the box owner — mirroring The Box. Denied = `Err`, not a
  warning.

### Host-safe ceiling (roadmap item 3)
`TheBox::new(path, ceiling_bytes)` or
`TheBox::with_host_fraction(path, 0.5)` (reads `/proc/meminfo`).
Opening a space that would push total reservations past the ceiling is
**refused, not degraded** — measured in the demo:

```
The Box: ceiling 64 MiB (host RAM 3 GiB)
Ceiling held: requested 50331648 B would push reservations past 33554432 B (ceiling 67108864 B)
```

### Acceptance criteria → measured/tested
- *Opening N spaces commits ~N×(touched segments)*: exact equality
  asserted — 3 spaces, 1 touched cell each = 3 × (256 KiB + 2 KiB).
- *Isolation*: same local cell, different values per space; untouched
  segments stay uncommitted; papers unreadable across spaces. Tested.
- *Ceiling enforced*: tested (`CeilingExceeded`), plus per-space
  `BudgetExceeded` as the ceiling's little sibling.
- *Disk boundary*: write/read past a planet's 65,536 blocks → `Err`. Tested.

## Build & run

```bash
rustc --edition=2021 -C opt-level=3 -C target-cpu=native cadran_vgpu.rs -o cadran_engine
./cadran_engine

rustc --edition=2021 --test cadran_vgpu.rs -o cadran_tests && ./cadran_tests   # 23 tests
```

## API sketch

```rust
let mut the_box = TheBox::with_host_fraction("cadran_box.disk", 0.5)?;
let team = the_box.open_space(16 << 20, Capabilities::default())?; // 16 MiB budget

let mut h = the_box.space(team)?;
h.write_cell(7, 0.8427, 17, 0);
let paper = h.put_paper(&pdf_bytes)?;
h.check_budget()?;                       // BudgetExceeded if over
h.require_network().unwrap_err();        // denied by default

the_box.release_space(team)?;            // RAM reclaimed, hole punched, budget back
```

## Invariants held (unchanged from v2)

Stable Rust only · powers of two, shift+mask · mask-don't-trap in RAM,
refuse-don't-redirect at the disk boundary · all `unsafe` interior ·
`Send` not `Sync`, one space one owner · one LLM shared · every number
measured · every claim asserted or tested.

## Honest framing

v3 governs memory, disk, and capabilities per team. It does **not** run
the model. Next is Stage 2: SIMD kernels (`gemv`, `rmsnorm`, `softmax`,
`silu`) and multithreading — the throughput frontier the forward pass
will stand on.
