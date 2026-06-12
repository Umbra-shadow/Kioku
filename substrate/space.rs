// space.rs — the Space bundle and The Box owner
// One space = (vRAM planet, disk planet, capabilities) under one id.
// The Box owns the universe, enforces the host-safe ceiling, and is the
// only door through which a space is opened or released.

use std::fmt;
use std::io;
use std::path::Path;

use crate::cadran_storage::{CadranDiskManager, DiskObjectHandle};
use crate::cadran_vram::{absolute_address, CadranVRAMManager, NUM_PLANETS};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    pub host_filesystem: bool,
    pub network: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpaceId(pub u32);

#[derive(Debug)]
pub enum BoxError {
    CeilingExceeded {
        requested: u64,
        reserved: u64,
        ceiling: u64,
    },
    BudgetExceeded {
        committed: u64,
        budget: u64,
    },
    CapabilityDenied(&'static str),
    NoFreePlanet,
    SpaceClosed(SpaceId),
    Io(io::Error),
}

impl fmt::Display for BoxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoxError::CeilingExceeded {
                requested,
                reserved,
                ceiling,
            } => write!(
                f,
                "host-safe ceiling: requested {} B would push reservations past {} B (ceiling {} B)",
                requested, reserved, ceiling
            ),
            BoxError::BudgetExceeded { committed, budget } => write!(
                f,
                "space budget exceeded: {} B committed of {} B budget",
                committed, budget
            ),
            BoxError::CapabilityDenied(cap) => {
                write!(f, "capability '{}' not granted to this space", cap)
            }
            BoxError::NoFreePlanet => write!(f, "no free planet in the box"),
            BoxError::SpaceClosed(id) => write!(f, "space {:?} is closed", id),
            BoxError::Io(e) => write!(f, "io: {}", e),
        }
    }
}

impl std::error::Error for BoxError {}

impl From<io::Error> for BoxError {
    fn from(e: io::Error) -> Self {
        BoxError::Io(e)
    }
}

struct SpaceRecord {
    open: bool,
    budget_bytes: u64,
    capabilities: Capabilities,
}

pub struct TheBox {
    vram: CadranVRAMManager,
    disk: CadranDiskManager,
    ceiling_bytes: u64,
    reserved_bytes: u64,
    spaces: Vec<SpaceRecord>, // index == planet id; planet 0 is the lobby, never a space
}

/// Total host RAM (Linux /proc/meminfo). Falls back to 1 GiB if unreadable.
pub fn host_ram_bytes() -> u64 {
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest
                    .trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse()
                    .unwrap_or(0);
                if kb > 0 {
                    return kb * 1024;
                }
            }
        }
    }
    1 << 30
}

impl TheBox {
    pub fn new<P: AsRef<Path>>(disk_path: P, ceiling_bytes: u64) -> Result<Self, BoxError> {
        let mut spaces = Vec::with_capacity(64);
        spaces.push(SpaceRecord {
            open: false,
            budget_bytes: 0,
            capabilities: Capabilities::default(),
        }); // lobby
        Ok(TheBox {
            vram: CadranVRAMManager::construct_universe(),
            disk: CadranDiskManager::open(disk_path)?,
            ceiling_bytes,
            reserved_bytes: 0,
            spaces,
        })
    }

    /// Ceiling as a fraction of physical host RAM.
    pub fn with_host_fraction<P: AsRef<Path>>(
        disk_path: P,
        fraction: f64,
    ) -> Result<Self, BoxError> {
        let frac = fraction.clamp(0.01, 0.95);
        Self::new(disk_path, (host_ram_bytes() as f64 * frac) as u64)
    }

    /// Opens a space: reserves its memory budget against the ceiling and
    /// assigns it a planet. Refused — not degraded — when the ceiling
    /// would be crossed.
    pub fn open_space(
        &mut self,
        budget_bytes: u64,
        capabilities: Capabilities,
    ) -> Result<SpaceId, BoxError> {
        if self.reserved_bytes + budget_bytes > self.ceiling_bytes {
            return Err(BoxError::CeilingExceeded {
                requested: budget_bytes,
                reserved: self.reserved_bytes,
                ceiling: self.ceiling_bytes,
            });
        }
        let planet = match (1..self.spaces.len()).find(|&p| !self.spaces[p].open) {
            Some(p) => p,
            None => {
                if self.spaces.len() >= NUM_PLANETS {
                    return Err(BoxError::NoFreePlanet);
                }
                self.spaces.push(SpaceRecord {
                    open: false,
                    budget_bytes: 0,
                    capabilities: Capabilities::default(),
                });
                self.spaces.len() - 1
            }
        };
        self.spaces[planet] = SpaceRecord {
            open: true,
            budget_bytes,
            capabilities,
        };
        self.reserved_bytes += budget_bytes;
        Ok(SpaceId(planet as u32))
    }

    fn record(&self, id: SpaceId) -> Result<&SpaceRecord, BoxError> {
        match self.spaces.get(id.0 as usize) {
            Some(r) if r.open => Ok(r),
            _ => Err(BoxError::SpaceClosed(id)),
        }
    }

    /// Borrow a handle scoped to one space's planet. While the handle
    /// lives, nothing else can touch the box — one space, one owner.
    pub fn space(&mut self, id: SpaceId) -> Result<SpaceHandle<'_>, BoxError> {
        self.record(id)?;
        Ok(SpaceHandle { the_box: self, id })
    }

    /// Tears the whole space down: vRAM planet reclaimed, disk hole
    /// punched, budget reservation returned. Returns RAM bytes freed.
    pub fn release_space(&mut self, id: SpaceId) -> Result<u64, BoxError> {
        self.record(id)?;
        let planet = id.0 as usize;
        let freed = self.vram.release_planet(planet);
        self.disk.release_planet(planet)?;
        self.reserved_bytes -= self.spaces[planet].budget_bytes;
        self.spaces[planet] = SpaceRecord {
            open: false,
            budget_bytes: 0,
            capabilities: Capabilities::default(),
        };
        Ok(freed)
    }

    pub fn committed_bytes(&self) -> u64 {
        self.vram.committed_bytes()
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }

    pub fn ceiling_bytes(&self) -> u64 {
        self.ceiling_bytes
    }

    pub fn open_spaces(&self) -> usize {
        self.spaces.iter().filter(|s| s.open).count()
    }
}

pub struct SpaceHandle<'b> {
    the_box: &'b mut TheBox,
    id: SpaceId,
}

impl<'b> SpaceHandle<'b> {
    #[inline(always)]
    fn planet(&self) -> usize {
        self.id.0 as usize
    }

    #[inline(always)]
    pub fn write_cell(&mut self, local_cell: u64, activation: f32, expert: u32, weight: u64) {
        let c = self
            .the_box
            .vram
            .access(absolute_address(self.planet(), local_cell));
        c.unquantized_activation = activation;
        c.active_expert_id = expert;
        c.weight_pointer = weight;
    }

    #[inline(always)]
    pub fn read_cell(&mut self, local_cell: u64) -> f32 {
        self.the_box
            .vram
            .access(absolute_address(self.planet(), local_cell))
            .unquantized_activation
    }

    /// Non-committing read: None if this space never touched the cell.
    pub fn peek_cell(&self, local_cell: u64) -> Option<f32> {
        self.the_box
            .vram
            .peek(absolute_address(self.planet(), local_cell))
            .map(|c| c.unquantized_activation)
    }

    /// Persist one artifact (a paper, a PDF) in this space's disk room.
    pub fn put_paper(&mut self, payload: &[u8]) -> Result<DiskObjectHandle, BoxError> {
        Ok(self.the_box.disk.put_object(self.planet(), payload)?)
    }

    pub fn get_paper(&mut self, handle: DiskObjectHandle) -> Result<Vec<u8>, BoxError> {
        if handle.planet_id as usize != self.planet() {
            return Err(BoxError::CapabilityDenied("cross-space paper access"));
        }
        Ok(self.the_box.disk.get_object(handle)?)
    }

    /// RAM this space has actually committed (segments + its table).
    pub fn committed_bytes(&self) -> u64 {
        self.the_box.vram.planet_committed_bytes(self.planet())
    }

    /// The ceiling's little sibling: a space must live within its budget.
    pub fn check_budget(&self) -> Result<(), BoxError> {
        let committed = self.committed_bytes();
        let budget = self.the_box.spaces[self.planet()].budget_bytes;
        if committed > budget {
            Err(BoxError::BudgetExceeded { committed, budget })
        } else {
            Ok(())
        }
    }

    pub fn require_network(&self) -> Result<(), BoxError> {
        if self.the_box.spaces[self.planet()].capabilities.network {
            Ok(())
        } else {
            Err(BoxError::CapabilityDenied("network"))
        }
    }

    pub fn require_host_filesystem(&self) -> Result<(), BoxError> {
        if self.the_box.spaces[self.planet()].capabilities.host_filesystem {
            Ok(())
        } else {
            Err(BoxError::CapabilityDenied("host_filesystem"))
        }
    }

    pub fn id(&self) -> SpaceId {
        self.id
    }
}

// ---------------------------------------------------------------------------
// Kioku v1 extensions — additive only; nothing above this line changed.
// kiokud (the memory daemon) needs full-cell reads and substrate-wide gauges
// that the original surface did not expose. `allow(dead_code)` because the
// original cadran_vgpu build does not call these.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
impl TheBox {
    /// Every open space id, lowest planet first.
    pub fn open_space_ids(&self) -> Vec<SpaceId> {
        (1..self.spaces.len())
            .filter(|&p| self.spaces[p].open)
            .map(|p| SpaceId(p as u32))
            .collect()
    }

    /// The budget a space was opened with.
    pub fn space_budget_bytes(&self, id: SpaceId) -> Result<u64, BoxError> {
        Ok(self.record(id)?.budget_bytes)
    }

    /// Real bytes the host filesystem has committed for the 4 TiB sparse disk.
    pub fn disk_committed_bytes(&self) -> Result<u64, BoxError> {
        Ok(self.disk.committed_bytes()?)
    }

    pub const fn disk_virtual_bytes(&self) -> u64 {
        crate::cadran_storage::DISK_VIRTUAL_BYTES
    }

    pub const fn vram_virtual_bytes(&self) -> u64 {
        crate::cadran_vram::UNIVERSE_BYTES
    }
}

#[allow(dead_code)]
impl<'b> SpaceHandle<'b> {
    /// Non-committing full-cell read: (activation, expert, weight).
    /// None if this space never touched the cell.
    pub fn peek_cell_full(&self, local_cell: u64) -> Option<(f32, u32, u64)> {
        self.the_box
            .vram
            .peek(absolute_address(self.planet(), local_cell))
            .map(|c| (c.unquantized_activation, c.active_expert_id, c.weight_pointer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cadran_vram::SEGMENT_BYTES;

    fn temp_disk(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("cadran_box_test_{}_{}", std::process::id(), name));
        p
    }

    fn cleanup(p: &std::path::Path) {
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn spaces_are_isolated() {
        let path = temp_disk("iso");
        let mut b = TheBox::new(&path, 64 << 20).unwrap();
        let s1 = b.open_space(8 << 20, Capabilities::default()).unwrap();
        let s2 = b.open_space(8 << 20, Capabilities::default()).unwrap();
        b.space(s1).unwrap().write_cell(7, 1.5, 0, 0);
        b.space(s2).unwrap().write_cell(7, 9.5, 0, 0);
        assert_eq!(b.space(s1).unwrap().read_cell(7), 1.5);
        assert_eq!(b.space(s2).unwrap().read_cell(7), 9.5);
        // a segment s2 never touched stays uncommitted
        assert_eq!(b.space(s2).unwrap().peek_cell(1 << 21), None);
        cleanup(&path);
    }

    #[test]
    fn ceiling_refuses_not_degrades() {
        let path = temp_disk("ceiling");
        let mut b = TheBox::new(&path, 10 << 20).unwrap();
        let _s1 = b.open_space(8 << 20, Capabilities::default()).unwrap();
        match b.open_space(8 << 20, Capabilities::default()) {
            Err(BoxError::CeilingExceeded { .. }) => {}
            other => panic!("expected CeilingExceeded, got {:?}", other.map(|s| s.0)),
        }
        cleanup(&path);
    }

    #[test]
    fn release_returns_reservation_and_planet() {
        let path = temp_disk("release");
        let mut b = TheBox::new(&path, 10 << 20).unwrap();
        let s1 = b.open_space(8 << 20, Capabilities::default()).unwrap();
        b.space(s1).unwrap().write_cell(0, 1.0, 0, 0);
        assert!(b.committed_bytes() > 0);
        b.release_space(s1).unwrap();
        assert_eq!(b.reserved_bytes(), 0);
        let s2 = b.open_space(8 << 20, Capabilities::default()).unwrap();
        assert_eq!(s2.0, s1.0); // planet reused
        assert_eq!(b.space(s2).unwrap().peek_cell(0), None); // and clean
        cleanup(&path);
    }

    #[test]
    fn n_spaces_commit_n_touched_segments() {
        let path = temp_disk("commit");
        let mut b = TheBox::new(&path, 64 << 20).unwrap();
        let base = b.committed_bytes();
        let mut ids = Vec::new();
        for _ in 0..3 {
            let s = b.open_space(4 << 20, Capabilities::default()).unwrap();
            b.space(s).unwrap().write_cell(0, 1.0, 0, 0);
            ids.push(s);
        }
        let per_space = SEGMENT_BYTES + 2048; // one segment + planet table
        assert_eq!(b.committed_bytes() - base, 3 * per_space);
        cleanup(&path);
    }

    #[test]
    fn budget_is_enforced() {
        let path = temp_disk("budget");
        let mut b = TheBox::new(&path, 64 << 20).unwrap();
        let s = b
            .open_space(SEGMENT_BYTES + 4096, Capabilities::default())
            .unwrap();
        let mut h = b.space(s).unwrap();
        h.write_cell(0, 1.0, 0, 0);
        assert!(h.check_budget().is_ok());
        h.write_cell(10 * SEGMENT_BYTES / 16, 1.0, 0, 0); // second segment
        match h.check_budget() {
            Err(BoxError::BudgetExceeded { .. }) => {}
            other => panic!("expected BudgetExceeded, got {:?}", other),
        }
        cleanup(&path);
    }

    #[test]
    fn capabilities_default_off() {
        let path = temp_disk("caps");
        let mut b = TheBox::new(&path, 64 << 20).unwrap();
        let s_off = b.open_space(1 << 20, Capabilities::default()).unwrap();
        let s_on = b
            .open_space(
                1 << 20,
                Capabilities {
                    network: true,
                    host_filesystem: false,
                },
            )
            .unwrap();
        assert!(b.space(s_off).unwrap().require_network().is_err());
        assert!(b.space(s_off).unwrap().require_host_filesystem().is_err());
        assert!(b.space(s_on).unwrap().require_network().is_ok());
        assert!(b.space(s_on).unwrap().require_host_filesystem().is_err());
        cleanup(&path);
    }

    #[test]
    fn papers_stay_in_their_space() {
        let path = temp_disk("papers");
        let mut b = TheBox::new(&path, 64 << 20).unwrap();
        let s1 = b.open_space(1 << 20, Capabilities::default()).unwrap();
        let s2 = b.open_space(1 << 20, Capabilities::default()).unwrap();
        let h = b.space(s1).unwrap().put_paper(b"team one findings").unwrap();
        assert_eq!(
            b.space(s1).unwrap().get_paper(h).unwrap(),
            b"team one findings"
        );
        assert!(b.space(s2).unwrap().get_paper(h).is_err());
        cleanup(&path);
    }
}
