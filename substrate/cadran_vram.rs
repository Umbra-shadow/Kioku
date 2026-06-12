// cadran_vram.rs — Cadran virtual RAM
// Stable Rust. Build: rustc -C opt-level=3 cadran_vgpu.rs -o cadran_engine

use std::alloc::{alloc_zeroed, dealloc, Layout};

pub const PLANET_BITS: usize = 14; // 16,384 planets (one per space/team)
pub const PLANET_CELL_BITS: usize = 22; // 4,194,304 cells per planet
pub const SEGMENT_CELL_BITS: usize = 14; // 16,384 cells per segment (fault granularity)

pub const NUM_PLANETS: usize = 1 << PLANET_BITS;
pub const CELLS_PER_PLANET: usize = 1 << PLANET_CELL_BITS;
pub const CELLS_PER_SEGMENT: usize = 1 << SEGMENT_CELL_BITS;
pub const SEGMENTS_PER_PLANET: usize = CELLS_PER_PLANET / CELLS_PER_SEGMENT;

pub const UNIVERSE_CELL_BITS: usize = PLANET_BITS + PLANET_CELL_BITS; // 36
pub const UNIVERSE_CELLS: u64 = 1u64 << UNIVERSE_CELL_BITS;
pub const UNIVERSE_CELL_MASK: u64 = UNIVERSE_CELLS - 1;
pub const PLANET_CELL_MASK: u64 = (CELLS_PER_PLANET - 1) as u64;
pub const SEGMENT_CELL_MASK: u64 = (CELLS_PER_SEGMENT - 1) as u64;

pub const CELL_BYTES: usize = std::mem::size_of::<VRAMCell>();
pub const UNIVERSE_BYTES: u64 = UNIVERSE_CELLS * CELL_BYTES as u64;
pub const SEGMENT_BYTES: u64 = (CELLS_PER_SEGMENT * CELL_BYTES) as u64;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct VRAMCell {
    pub unquantized_activation: f32,
    pub active_expert_id: u32,
    pub weight_pointer: u64,
}

const _: () = assert!(CELL_BYTES == 16);
const _: () = assert!(UNIVERSE_BYTES == 1u64 << 40); // exactly 1 TiB, claim is now true
const _: () = assert!(SEGMENT_BYTES == 256 * 1024);
const _: () = assert!(SEGMENTS_PER_PLANET == 256);

#[repr(C, align(64))]
pub struct Segment {
    pub cells: [VRAMCell; CELLS_PER_SEGMENT],
}

#[repr(C)]
struct SegmentTable {
    segments: [*mut Segment; SEGMENTS_PER_PLANET],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CadranStats {
    pub planets_resident: u64,
    pub segments_resident: u64,
    pub planet_faults: u64,
    pub segment_faults: u64,
}

pub struct CadranVRAMManager {
    directory: Box<[*mut SegmentTable]>,
    stats: CadranStats,
}

// Sound: the manager exclusively owns every allocation it points to.
// Deliberately NOT Sync — one space, one owner; cross-team sharing goes
// through the Necklace, never through aliased memory.
unsafe impl Send for CadranVRAMManager {}

#[inline(always)]
const fn split(addr: u64) -> (usize, usize, usize) {
    let addr = addr & UNIVERSE_CELL_MASK;
    (
        (addr >> PLANET_CELL_BITS) as usize,
        ((addr >> SEGMENT_CELL_BITS) as usize) & (SEGMENTS_PER_PLANET - 1),
        (addr & SEGMENT_CELL_MASK) as usize,
    )
}

#[inline(always)]
pub const fn absolute_address(planet_id: usize, local_cell: u64) -> u64 {
    (((planet_id & (NUM_PLANETS - 1)) as u64) << PLANET_CELL_BITS)
        | (local_cell & PLANET_CELL_MASK)
}

impl CadranVRAMManager {
    pub fn construct_universe() -> Self {
        CadranVRAMManager {
            directory: vec![std::ptr::null_mut(); NUM_PLANETS].into_boxed_slice(),
            stats: CadranStats::default(),
        }
    }

    #[cold]
    #[inline(never)]
    fn fault_planet(&mut self, planet_id: usize) -> *mut SegmentTable {
        let layout = Layout::new::<SegmentTable>();
        let table = unsafe { alloc_zeroed(layout) } as *mut SegmentTable;
        if table.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        self.directory[planet_id] = table;
        self.stats.planets_resident += 1;
        self.stats.planet_faults += 1;
        table
    }

    #[cold]
    #[inline(never)]
    fn fault_segment(&mut self, table: *mut SegmentTable, seg_id: usize) -> *mut Segment {
        let layout = Layout::new::<Segment>();
        let seg = unsafe { alloc_zeroed(layout) } as *mut Segment;
        if seg.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        unsafe { (*table).segments[seg_id] = seg };
        self.stats.segments_resident += 1;
        self.stats.segment_faults += 1;
        seg
    }

    /// Faulting segment lookup. Commits a 256 KiB segment (and a 2 KiB
    /// planet table) only on first touch — "small outside, huge inside".
    #[inline(always)]
    pub fn segment_ptr(&mut self, planet_id: usize, seg_id: usize) -> *mut Segment {
        let planet_id = planet_id & (NUM_PLANETS - 1);
        let seg_id = seg_id & (SEGMENTS_PER_PLANET - 1);
        unsafe {
            let mut table = *self.directory.get_unchecked(planet_id);
            if table.is_null() {
                table = self.fault_planet(planet_id);
            }
            let mut seg = *(*table).segments.get_unchecked(seg_id);
            if seg.is_null() {
                seg = self.fault_segment(table, seg_id);
            }
            seg
        }
    }

    /// Hot-path cell pointer. Address is masked to the universe, so a bad
    /// index can never escape into foreign memory; it can only wrap.
    ///
    /// # Safety
    /// The returned pointer is valid until the planet is released or the
    /// manager is dropped. Caller must not hold it across either.
    #[inline(always)]
    pub unsafe fn cell_ptr(&mut self, absolute_cell: u64) -> *mut VRAMCell {
        let (planet_id, seg_id, offset) = split(absolute_cell);
        let seg = self.segment_ptr(planet_id, seg_id);
        (*seg).cells.as_mut_ptr().add(offset)
    }

    /// Safe faulting access.
    #[inline(always)]
    pub fn access(&mut self, absolute_cell: u64) -> &mut VRAMCell {
        unsafe { &mut *self.cell_ptr(absolute_cell) }
    }

    /// Non-faulting read: never commits memory.
    pub fn peek(&self, absolute_cell: u64) -> Option<VRAMCell> {
        let (planet_id, seg_id, offset) = split(absolute_cell);
        let table = self.directory[planet_id];
        if table.is_null() {
            return None;
        }
        let seg = unsafe { (*table).segments[seg_id] };
        if seg.is_null() {
            return None;
        }
        Some(unsafe { (*seg).cells[offset] })
    }

    pub fn prefault_range(&mut self, planet_id: usize, first_cell: u64, len: usize) {
        if len == 0 {
            return;
        }
        let first_seg = ((first_cell & PLANET_CELL_MASK) >> SEGMENT_CELL_BITS) as usize;
        let last_seg = (((first_cell + (len as u64 - 1)) & PLANET_CELL_MASK)
            >> SEGMENT_CELL_BITS) as usize;
        let mut s = first_seg;
        loop {
            self.segment_ptr(planet_id, s);
            if s == last_seg {
                break;
            }
            s = (s + 1) & (SEGMENTS_PER_PLANET - 1);
        }
    }

    /// Tear down one team's room and reclaim its committed bytes.
    pub fn release_planet(&mut self, planet_id: usize) -> u64 {
        let planet_id = planet_id & (NUM_PLANETS - 1);
        let table = self.directory[planet_id];
        if table.is_null() {
            return 0;
        }
        let mut freed = std::mem::size_of::<SegmentTable>() as u64;
        unsafe {
            for s in 0..SEGMENTS_PER_PLANET {
                let seg = (*table).segments[s];
                if !seg.is_null() {
                    dealloc(seg as *mut u8, Layout::new::<Segment>());
                    freed += SEGMENT_BYTES;
                    self.stats.segments_resident -= 1;
                }
            }
            dealloc(table as *mut u8, Layout::new::<SegmentTable>());
        }
        self.directory[planet_id] = std::ptr::null_mut();
        self.stats.planets_resident -= 1;
        freed
    }

    pub fn stats(&self) -> CadranStats {
        self.stats
    }

    /// RAM one planet has actually committed (its segments + table).
    pub fn planet_committed_bytes(&self, planet_id: usize) -> u64 {
        let table = self.directory[planet_id & (NUM_PLANETS - 1)];
        if table.is_null() {
            return 0;
        }
        let mut segs = 0u64;
        unsafe {
            for s in 0..SEGMENTS_PER_PLANET {
                if !(*table).segments[s].is_null() {
                    segs += 1;
                }
            }
        }
        segs * SEGMENT_BYTES + std::mem::size_of::<SegmentTable>() as u64
    }

    pub fn committed_bytes(&self) -> u64 {
        self.stats.segments_resident * SEGMENT_BYTES
            + self.stats.planets_resident * std::mem::size_of::<SegmentTable>() as u64
            + (NUM_PLANETS * std::mem::size_of::<*mut SegmentTable>()) as u64
    }

    pub const fn virtual_bytes(&self) -> u64 {
        UNIVERSE_BYTES
    }
}

impl Drop for CadranVRAMManager {
    fn drop(&mut self) {
        for p in 0..NUM_PLANETS {
            let table = self.directory[p];
            if table.is_null() {
                continue;
            }
            unsafe {
                for s in 0..SEGMENTS_PER_PLANET {
                    let seg = (*table).segments[s];
                    if !seg.is_null() {
                        dealloc(seg as *mut u8, Layout::new::<Segment>());
                    }
                }
                dealloc(table as *mut u8, Layout::new::<SegmentTable>());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_split_roundtrip() {
        let addr = absolute_address(5, 0x12_3456);
        let (p, s, o) = split(addr);
        assert_eq!(p, 5);
        assert_eq!(s, 0x12_3456 >> SEGMENT_CELL_BITS);
        assert_eq!(o, 0x12_3456 & SEGMENT_CELL_MASK as usize);
    }

    #[test]
    fn fault_on_first_touch_only() {
        let mut m = CadranVRAMManager::construct_universe();
        assert_eq!(m.committed_bytes(), (NUM_PLANETS * 8) as u64);
        let a = absolute_address(7, 42);
        m.access(a).unquantized_activation = 1.5;
        let st = m.stats();
        assert_eq!(st.planet_faults, 1);
        assert_eq!(st.segment_faults, 1);
        m.access(a + 1).unquantized_activation = 2.5;
        assert_eq!(m.stats().segment_faults, 1);
        assert_eq!(m.peek(a).unwrap().unquantized_activation, 1.5);
        assert!(m.peek(absolute_address(8, 0)).is_none());
    }

    #[test]
    fn zeroed_on_commit() {
        let mut m = CadranVRAMManager::construct_universe();
        let c = *m.access(absolute_address(3, 999));
        assert_eq!(c.unquantized_activation, 0.0);
        assert_eq!(c.active_expert_id, 0);
        assert_eq!(c.weight_pointer, 0);
    }

    #[test]
    fn release_reclaims() {
        let mut m = CadranVRAMManager::construct_universe();
        m.access(absolute_address(2, 0));
        m.access(absolute_address(2, CELLS_PER_SEGMENT as u64));
        let base = (NUM_PLANETS * 8) as u64;
        assert_eq!(m.committed_bytes(), base + 2 * SEGMENT_BYTES + 2048);
        let freed = m.release_planet(2);
        assert_eq!(freed, 2 * SEGMENT_BYTES + 2048);
        assert_eq!(m.committed_bytes(), base);
        assert!(m.peek(absolute_address(2, 0)).is_none());
    }

    #[test]
    fn bad_index_cannot_escape() {
        let mut m = CadranVRAMManager::construct_universe();
        m.access(u64::MAX).unquantized_activation = 9.0;
        let wrapped = u64::MAX & UNIVERSE_CELL_MASK;
        assert_eq!(m.peek(wrapped).unwrap().unquantized_activation, 9.0);
    }
}
