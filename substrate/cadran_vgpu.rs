// cadran_vgpu.rs — Cadran virtual GPU
// Build: rustc -C opt-level=3 -C target-cpu=native cadran_vgpu.rs -o cadran_engine
// Run:   ./cadran_engine
// Test:  rustc --test cadran_vgpu.rs -o cadran_tests && ./cadran_tests

use std::hint::black_box;
use std::time::Instant;

mod cadran_storage;
mod cadran_vram;
mod space;
use cadran_vram::{
    absolute_address, CadranStats, CadranVRAMManager, PLANET_CELL_MASK, SEGMENT_CELL_BITS,
    SEGMENT_CELL_MASK, CELLS_PER_SEGMENT, UNIVERSE_BYTES,
};

pub const LANES: usize = 4;

/// Borrows the vRAM for its whole lifetime: lanes can never dangle, and no
/// one else can mutate the universe while the vGPU is bound to it.
pub struct CadranVGpuCore<'u> {
    vram: &'u mut CadranVRAMManager,
    lane_planets: [usize; LANES],
}

impl<'u> CadranVGpuCore<'u> {
    pub fn bind(vram: &'u mut CadranVRAMManager, lane_planets: [usize; LANES]) -> Self {
        for planet in lane_planets {
            vram.segment_ptr(planet, 0);
        }
        CadranVGpuCore { vram, lane_planets }
    }

    #[inline(always)]
    fn lane_addr(&self, lane: usize, cell: u64) -> u64 {
        absolute_address(self.lane_planets[lane & (LANES - 1)], cell)
    }

    #[inline(always)]
    pub fn write(&mut self, lane: usize, cell: u64, activation: f32, expert: u32, weight: u64) {
        let c = self.vram.access(self.lane_addr(lane, cell));
        c.unquantized_activation = activation;
        c.active_expert_id = expert;
        c.weight_pointer = weight;
    }

    #[inline(always)]
    pub fn read(&mut self, lane: usize, cell: u64) -> f32 {
        self.vram
            .access(self.lane_addr(lane, cell))
            .unquantized_activation
    }

    /// Compatibility gather: two packed u64 registers -> 4 lane reads.
    /// idx_n = (a_n << 8) | b_n, one 16-bit index per lane.
    #[inline(always)]
    pub fn process_vector_pass(&mut self, packed_reg_a: u64, packed_reg_b: u64) -> [f32; LANES] {
        let mut out = [0.0f32; LANES];
        let mut lane = 0;
        while lane < LANES {
            let shift = (24 - 8 * lane) as u64;
            let a = (packed_reg_a >> shift) & 0xFF;
            let b = (packed_reg_b >> shift) & 0xFF;
            out[lane] = self.read(lane, (a << 8) | b);
            lane += 1;
        }
        out
    }

    fn for_each_pair_chunk<F>(&mut self, lane: usize, a: u64, b: u64, len: usize, mut f: F)
    where
        F: FnMut(*mut cadran_vram::VRAMCell, *mut cadran_vram::VRAMCell, usize),
    {
        let planet = self.lane_planets[lane & (LANES - 1)];
        let mut done = 0usize;
        while done < len {
            let ac = (a + done as u64) & PLANET_CELL_MASK;
            let bc = (b + done as u64) & PLANET_CELL_MASK;
            let a_room = CELLS_PER_SEGMENT - (ac & SEGMENT_CELL_MASK) as usize;
            let b_room = CELLS_PER_SEGMENT - (bc & SEGMENT_CELL_MASK) as usize;
            let n = (len - done).min(a_room).min(b_room);
            let sa = self.vram.segment_ptr(planet, (ac >> SEGMENT_CELL_BITS) as usize);
            let sb = self.vram.segment_ptr(planet, (bc >> SEGMENT_CELL_BITS) as usize);
            unsafe {
                let pa = (*sa).cells.as_mut_ptr().add((ac & SEGMENT_CELL_MASK) as usize);
                let pb = (*sb).cells.as_mut_ptr().add((bc & SEGMENT_CELL_MASK) as usize);
                f(pa, pb, n);
            }
            done += n;
        }
    }

    /// Streaming dot product over activations of one lane's planet.
    pub fn dot(&mut self, lane: usize, a_base: u64, b_base: u64, len: usize) -> f32 {
        let mut acc = [0.0f32; 8];
        self.for_each_pair_chunk(lane, a_base, b_base, len, |pa, pb, n| unsafe {
            let mut j = 0usize;
            while j + 8 <= n {
                let mut k = 0;
                while k < 8 {
                    acc[k] += (*pa.add(j + k)).unquantized_activation
                        * (*pb.add(j + k)).unquantized_activation;
                    k += 1;
                }
                j += 8;
            }
            while j < n {
                acc[0] += (*pa.add(j)).unquantized_activation
                    * (*pb.add(j)).unquantized_activation;
                j += 1;
            }
        });
        acc.iter().sum()
    }

    /// y = alpha * x + y, in place, over one lane's planet.
    pub fn axpy(&mut self, lane: usize, alpha: f32, x_base: u64, y_base: u64, len: usize) {
        self.for_each_pair_chunk(lane, x_base, y_base, len, |px, py, n| unsafe {
            let mut j = 0usize;
            while j < n {
                let y = &mut (*py.add(j)).unquantized_activation;
                *y = alpha * (*px.add(j)).unquantized_activation + *y;
                j += 1;
            }
        });
    }

    /// Deterministic fill of a lane range (also measures commit+write speed).
    pub fn fill(&mut self, lane: usize, base: u64, len: usize, scale: f32) {
        self.for_each_pair_chunk(lane, base, base, len, |pa, _, n| unsafe {
            let mut j = 0usize;
            while j < n {
                (*pa.add(j)).unquantized_activation = (j as f32 + 1.0) * scale;
                j += 1;
            }
        });
    }

    pub fn stats(&self) -> CadranStats {
        self.vram.stats()
    }

    pub fn committed_bytes(&self) -> u64 {
        self.vram.committed_bytes()
    }
}

fn main() {
    println!("=== CADRAN VIRTUAL HARDWARE: vRAM + vGPU ===");
    let mut vram = CadranVRAMManager::construct_universe();
    println!(
        "Virtual universe: {} GiB ({} bytes), committed at boot: {} KiB",
        UNIVERSE_BYTES >> 30,
        UNIVERSE_BYTES,
        vram.committed_bytes() >> 10
    );

    let mut gpu = CadranVGpuCore::bind(&mut vram, [1, 2, 3, 4]);

    gpu.write(0, 42, 0.8427, 17, 0xDEAD_BEEF);
    let reg_a: u64 = 0;
    let reg_b: u64 = 42 << 24;
    let probe = gpu.process_vector_pass(reg_a, reg_b);
    assert_eq!(probe[0], 0.8427);
    println!("Gather verification: {:?}", probe);

    let gather_iters: u64 = 10_000_000;
    let start = Instant::now();
    let mut out = [0.0f32; LANES];
    for _ in 0..gather_iters {
        out = gpu.process_vector_pass(black_box(reg_a), black_box(reg_b));
    }
    let dt = start.elapsed();
    println!(
        "Gather pass: {} iters, {:.3} ns/lane-read ({:?} sample)",
        gather_iters,
        dt.as_nanos() as f64 / (gather_iters * LANES as u64) as f64,
        out
    );

    let len: usize = 1 << 20;
    let a_base: u64 = 0;
    let b_base: u64 = (len as u64) + (1 << SEGMENT_CELL_BITS);

    let start = Instant::now();
    gpu.fill(0, a_base, len, 1e-6);
    gpu.fill(0, b_base, len, 2e-6);
    let dt = start.elapsed();
    let fill_bytes = (2 * len * cadran_vram::CELL_BYTES) as f64;
    println!(
        "Fill (commit + write): {:.2} GiB/s over {} cells",
        fill_bytes / dt.as_secs_f64() / (1u64 << 30) as f64,
        2 * len
    );

    let dot_reps: u32 = 64;
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..dot_reps {
        sink += gpu.dot(0, black_box(a_base), black_box(b_base), len);
    }
    let dt = start.elapsed();
    let flops = 2.0 * len as f64 * dot_reps as f64;
    println!(
        "Dot kernel: {:.2} GFLOP/s (len {} x {} reps, sink {:.3})",
        flops / dt.as_secs_f64() / 1e9,
        len,
        dot_reps,
        sink
    );

    let start = Instant::now();
    for _ in 0..dot_reps {
        gpu.axpy(0, black_box(1.0000001f32), a_base, b_base, len);
    }
    let dt = start.elapsed();
    let axpy_bytes = (3 * len * cadran_vram::CELL_BYTES) as f64 * dot_reps as f64;
    println!(
        "AXPY kernel: {:.2} GiB/s effective traffic",
        axpy_bytes / dt.as_secs_f64() / (1u64 << 30) as f64
    );

    let st = gpu.stats();
    let committed = gpu.committed_bytes();
    println!("---------------------------------------------------------------");
    println!(
        "Planets resident: {} | Segments resident: {} | Faults: {}p/{}s",
        st.planets_resident, st.segments_resident, st.planet_faults, st.segment_faults
    );
    println!(
        "Committed: {:.2} MiB of {} GiB virtual ({:.6}%) — small outside, huge inside",
        committed as f64 / (1u64 << 20) as f64,
        UNIVERSE_BYTES >> 30,
        committed as f64 / UNIVERSE_BYTES as f64 * 100.0
    );

    drop(gpu);
    let freed = vram.release_planet(1);
    println!(
        "Released planet 1: {:.2} MiB reclaimed, committed now {:.2} MiB",
        freed as f64 / (1u64 << 20) as f64,
        vram.committed_bytes() as f64 / (1u64 << 20) as f64
    );
    println!("---------------------------------------------------------------");
    let disk_path = std::env::temp_dir().join("cadran.disk");
    let mut disk = cadran_storage::CadranDiskManager::open(&disk_path).expect("disk open");
    println!(
        "Virtual disk: {} TiB sparse, committed before write: {} KiB",
        disk.virtual_bytes() >> 40,
        disk.committed_bytes().expect("meta") >> 10
    );
    let paper = vec![0xABu8; 1 << 20];
    let t = Instant::now();
    let handle = disk.put_object(1, &paper).expect("put");
    disk.sync().expect("sync");
    let put_dt = t.elapsed();
    let t = Instant::now();
    let back = disk.get_object(handle).expect("get");
    let get_dt = t.elapsed();
    assert_eq!(back, paper);
    println!(
        "Disk object: 1 MiB paper stored at planet {} block {} | put+fsync {:.2} ms, get+CRC {:.2} ms",
        handle.planet_id, handle.first_block,
        put_dt.as_secs_f64() * 1e3, get_dt.as_secs_f64() * 1e3
    );
    println!(
        "Disk committed after write: {} KiB of {} TiB ({} objects)",
        disk.committed_bytes().expect("meta") >> 10,
        disk.virtual_bytes() >> 40,
        disk.stats().objects_written
    );
    disk.write_block(2, 0x10, b"raw scratch lane").expect("blk w");
    let mut scratch = [0u8; 16];
    disk.read_block(2, 0x10, &mut scratch).expect("blk r");
    assert_eq!(&scratch, b"raw scratch lane");
    println!("Raw block I/O verified on planet 2 block 0x10");
    let reclaimed = disk.release_planet(1).expect("release");
    println!(
        "Released disk planet 1: {} KiB logical reclaim, committed now {} KiB",
        reclaimed >> 10,
        disk.committed_bytes().expect("meta") >> 10
    );
    let _ = std::fs::remove_file(disk_path);
    println!("---------------------------------------------------------------");
    let box_disk = std::env::temp_dir().join("cadran_box.disk");
    let ceiling: u64 = 64 << 20;
    let mut the_box = space::TheBox::new(&box_disk, ceiling).expect("box");
    println!(
        "The Box: ceiling {} MiB (host RAM {} GiB), spaces open: {}",
        ceiling >> 20,
        space::host_ram_bytes() >> 30,
        the_box.open_spaces()
    );
    let s1 = the_box
        .open_space(16 << 20, space::Capabilities::default())
        .expect("s1");
    let s2 = the_box
        .open_space(16 << 20, space::Capabilities::default())
        .expect("s2");
    the_box.space(s1).expect("h1").write_cell(7, 1.5, 0, 0);
    the_box.space(s2).expect("h2").write_cell(7, 9.5, 0, 0);
    let iso = (
        the_box.space(s1).expect("h1").read_cell(7),
        the_box.space(s2).expect("h2").read_cell(7),
    );
    assert_eq!(iso, (1.5, 9.5));
    let paper = the_box
        .space(s1)
        .expect("h1")
        .put_paper(b"continaut findings, team 1")
        .expect("paper");
    let read_back = the_box.space(s1).expect("h1").get_paper(paper).expect("get");
    assert_eq!(read_back, b"continaut findings, team 1");
    assert!(the_box.space(s2).expect("h2").get_paper(paper).is_err());
    assert_eq!(the_box.space(s2).expect("h2").peek_cell(1 << 21), None);
    println!(
        "Spaces {} & {} isolated (cell 7 = {:?}); team 1 paper at block {} stays team 1's",
        the_box.space(s1).expect("h1").id().0, s2.0, iso, paper.first_block
    );
    let h1 = the_box.space(s1).expect("h1");
    println!(
        "Space {} committed {} KiB of its 16 MiB budget (within budget: {}); net/fs denied by default: {}/{}",
        s1.0,
        h1.committed_bytes() >> 10,
        h1.check_budget().is_ok(),
        h1.require_network().is_err(),
        h1.require_host_filesystem().is_err()
    );
    match the_box.open_space(48 << 20, space::Capabilities::default()) {
        Err(e) => println!("Ceiling held: {}", e),
        Ok(_) => panic!("ceiling failed to hold"),
    }
    let freed = the_box.release_space(s2).expect("release");
    println!(
        "Released space {}: {} KiB RAM reclaimed, reserved now {} MiB of {} MiB",
        s2.0,
        freed >> 10,
        the_box.reserved_bytes() >> 20,
        the_box.ceiling_bytes() >> 20
    );
    println!(
        "Box-wide committed: {} KiB under {} MiB ceiling",
        the_box.committed_bytes() >> 10,
        the_box.ceiling_bytes() >> 20
    );
    let _ = std::fs::remove_file(box_disk);
    let frac_disk = std::env::temp_dir().join("cadran_box_frac.disk");
    let frac_box = space::TheBox::with_host_fraction(&frac_disk, 0.5).expect("frac box");
    println!(
        "Host-fraction box: ceiling {} GiB = 50% of host RAM",
        frac_box.ceiling_bytes() >> 30
    );
    drop(frac_box);
    let _ = std::fs::remove_file(frac_disk);
    println!("===============================================================");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_matches_write() {
        let mut vram = CadranVRAMManager::construct_universe();
        let mut gpu = CadranVGpuCore::bind(&mut vram, [1, 2, 3, 4]);
        gpu.write(0, 0x012A, 1.25, 0, 0);
        gpu.write(1, 0x0100, 2.5, 0, 0);
        let a = (0x01u64 << 24) | (0x01 << 16);
        let b = (0x2Au64 << 24) | (0x00 << 16);
        let out = gpu.process_vector_pass(a, b);
        assert_eq!(out[0], 1.25);
        assert_eq!(out[1], 2.5);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn dot_is_exact_on_small_input() {
        let mut vram = CadranVRAMManager::construct_universe();
        let mut gpu = CadranVGpuCore::bind(&mut vram, [1, 2, 3, 4]);
        for i in 0..100u64 {
            gpu.write(2, i, 1.0, 0, 0);
            gpu.write(2, 1000 + i, 2.0, 0, 0);
        }
        let d = gpu.dot(2, 0, 1000, 100);
        assert_eq!(d, 200.0);
    }

    #[test]
    fn dot_crosses_segment_boundaries() {
        let mut vram = CadranVRAMManager::construct_universe();
        let mut gpu = CadranVGpuCore::bind(&mut vram, [1, 2, 3, 4]);
        let start = CELLS_PER_SEGMENT as u64 - 5;
        for i in 0..10u64 {
            gpu.write(3, start + i, 3.0, 0, 0);
            gpu.write(3, 4 * CELLS_PER_SEGMENT as u64 + i, 1.0, 0, 0);
        }
        let d = gpu.dot(3, start, 4 * CELLS_PER_SEGMENT as u64, 10);
        assert_eq!(d, 30.0);
    }

    #[test]
    fn axpy_in_place() {
        let mut vram = CadranVRAMManager::construct_universe();
        let mut gpu = CadranVGpuCore::bind(&mut vram, [1, 2, 3, 4]);
        for i in 0..16u64 {
            gpu.write(0, i, 2.0, 0, 0);
            gpu.write(0, 100 + i, 1.0, 0, 0);
        }
        gpu.axpy(0, 0.5, 0, 100, 16);
        assert_eq!(gpu.read(0, 100), 2.0);
        assert_eq!(gpu.read(0, 115), 2.0);
        assert_eq!(gpu.read(0, 116), 0.0);
    }

    #[test]
    fn lanes_are_isolated_planets() {
        let mut vram = CadranVRAMManager::construct_universe();
        let mut gpu = CadranVGpuCore::bind(&mut vram, [1, 2, 3, 4]);
        gpu.write(0, 7, 9.0, 0, 0);
        assert_eq!(gpu.read(1, 7), 0.0);
        assert_eq!(gpu.read(0, 7), 9.0);
    }
}
