// cadran_storage.rs — Cadran virtual disk
// A fixed virtual block device carved out of one sparse host file,
// addressed the same flat way as the vRAM: planet -> block -> byte.
// Linux/Unix. Stable Rust, std only.

use std::fs::{File, OpenOptions};
use std::io::{self, Error, ErrorKind};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::Path;

use crate::cadran_vram::NUM_PLANETS;

pub const DISK_BLOCK_BYTES: u64 = 4096;
pub const DISK_PLANET_BLOCK_BITS: usize = 16;
pub const DISK_BLOCKS_PER_PLANET: u64 = 1 << DISK_PLANET_BLOCK_BITS;
pub const DISK_PLANET_BYTES: u64 = DISK_BLOCKS_PER_PLANET * DISK_BLOCK_BYTES; // 256 MiB
pub const DISK_VIRTUAL_BYTES: u64 = NUM_PLANETS as u64 * DISK_PLANET_BYTES; // 4 TiB

const SUPERBLOCK_MAGIC: u64 = 0x4341_4452_4449_534B; // "CADRDISK"
const SUPERBLOCK_VERSION: u32 = 2;
const OBJECT_MAGIC: u32 = 0x4F42_4A31; // "OBJ1"
const OBJECT_HEADER_BYTES: u64 = 16;
const FIRST_DATA_BLOCK: u64 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiskObjectHandle {
    pub planet_id: u32,
    pub first_block: u64,
    pub len: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskStats {
    pub objects_written: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub planets_opened: u64,
    pub planets_released: u64,
}

pub struct CadranDiskManager {
    file: File,
    cursors: Vec<u64>, // next free block per planet; 0 = superblock not loaded
    stats: DiskStats,
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

#[cfg(target_os = "linux")]
fn punch_hole(file: &File, offset: u64, len: u64) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i32;
    }
    const FALLOC_FL_KEEP_SIZE: i32 = 0x01;
    const FALLOC_FL_PUNCH_HOLE: i32 = 0x02;
    let r = unsafe {
        fallocate(
            file.as_raw_fd(),
            FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE,
            offset as i64,
            len as i64,
        )
    };
    if r == 0 {
        Ok(())
    } else {
        Err(Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
fn punch_hole(_file: &File, _offset: u64, _len: u64) -> io::Result<()> {
    Ok(()) // logical release only; physical reclaim is Linux fallocate
}

#[inline(always)]
const fn planet_base(planet_id: usize) -> u64 {
    (planet_id & (NUM_PLANETS - 1)) as u64 * DISK_PLANET_BYTES
}

#[inline(always)]
const fn block_offset(planet_id: usize, block: u64) -> u64 {
    planet_base(planet_id) + (block & (DISK_BLOCKS_PER_PLANET - 1)) * DISK_BLOCK_BYTES
}

impl CadranDiskManager {
    /// Opens (or creates) the disk: one sparse file, 4 TiB virtual,
    /// zero bytes committed until written — same law as the vRAM.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        if file.metadata()?.len() != DISK_VIRTUAL_BYTES {
            file.set_len(DISK_VIRTUAL_BYTES)?;
        }
        Ok(CadranDiskManager {
            file,
            cursors: vec![0u64; NUM_PLANETS],
            stats: DiskStats::default(),
        })
    }

    fn superblock_bytes(cursor: u64) -> [u8; 24] {
        let mut b = [0u8; 24];
        b[0..8].copy_from_slice(&SUPERBLOCK_MAGIC.to_le_bytes());
        b[8..12].copy_from_slice(&SUPERBLOCK_VERSION.to_le_bytes());
        b[16..24].copy_from_slice(&cursor.to_le_bytes());
        b
    }

    fn write_superblock(&mut self, planet_id: usize, cursor: u64) -> io::Result<()> {
        self.file
            .write_all_at(&Self::superblock_bytes(cursor), planet_base(planet_id))
    }

    fn cursor(&mut self, planet_id: usize) -> io::Result<u64> {
        let planet_id = planet_id & (NUM_PLANETS - 1);
        if self.cursors[planet_id] != 0 {
            return Ok(self.cursors[planet_id]);
        }
        let mut b = [0u8; 24];
        self.file.read_exact_at(&mut b, planet_base(planet_id))?;
        let magic = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let cursor = if magic == SUPERBLOCK_MAGIC {
            let c = u64::from_le_bytes(b[16..24].try_into().unwrap());
            if c < FIRST_DATA_BLOCK || c > DISK_BLOCKS_PER_PLANET {
                return Err(Error::new(ErrorKind::InvalidData, "corrupt superblock cursor"));
            }
            c
        } else {
            self.write_superblock(planet_id, FIRST_DATA_BLOCK)?;
            FIRST_DATA_BLOCK
        };
        self.cursors[planet_id] = cursor;
        self.stats.planets_opened += 1;
        Ok(cursor)
    }

    fn check_bounds(planet_id: usize, block: u64, len: usize) -> io::Result<()> {
        if planet_id >= NUM_PLANETS {
            return Err(Error::new(ErrorKind::InvalidInput, "refused: planet outside the box"));
        }
        if block >= DISK_BLOCKS_PER_PLANET {
            return Err(Error::new(ErrorKind::InvalidInput, "refused: block outside planet boundary"));
        }
        if len as u64 > DISK_BLOCK_BYTES {
            return Err(Error::new(ErrorKind::InvalidInput, "refused: exceeds block size"));
        }
        Ok(())
    }

    /// Raw block write inside one planet's region. A write past the planet
    /// boundary is refused — never silently redirected, never UB.
    pub fn write_block(&mut self, planet_id: usize, block: u64, data: &[u8]) -> io::Result<()> {
        Self::check_bounds(planet_id, block, data.len())?;
        self.file.write_all_at(data, block_offset(planet_id, block))?;
        self.stats.bytes_written += data.len() as u64;
        Ok(())
    }

    pub fn read_block(&mut self, planet_id: usize, block: u64, buf: &mut [u8]) -> io::Result<()> {
        Self::check_bounds(planet_id, block, buf.len())?;
        self.file.read_exact_at(buf, block_offset(planet_id, block))?;
        self.stats.bytes_read += buf.len() as u64;
        Ok(())
    }

    /// Stores one object (a paper, a PDF, any byte stream) in a planet:
    /// [magic u32][crc u32][len u64][payload], block-aligned start.
    pub fn put_object(&mut self, planet_id: usize, payload: &[u8]) -> io::Result<DiskObjectHandle> {
        let planet_id = planet_id & (NUM_PLANETS - 1);
        let cursor = self.cursor(planet_id)?;
        let total = OBJECT_HEADER_BYTES + payload.len() as u64;
        let blocks_needed = (total + DISK_BLOCK_BYTES - 1) / DISK_BLOCK_BYTES;
        if cursor + blocks_needed > DISK_BLOCKS_PER_PLANET {
            return Err(Error::new(
                ErrorKind::Other,
                "planet disk full (256 MiB room)",
            ));
        }
        let offset = block_offset(planet_id, cursor);
        let mut header = [0u8; OBJECT_HEADER_BYTES as usize];
        header[0..4].copy_from_slice(&OBJECT_MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&crc32(payload).to_le_bytes());
        header[8..16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        self.file.write_all_at(&header, offset)?;
        self.file.write_all_at(payload, offset + OBJECT_HEADER_BYTES)?;
        let new_cursor = cursor + blocks_needed;
        self.write_superblock(planet_id, new_cursor)?;
        self.cursors[planet_id] = new_cursor;
        self.stats.objects_written += 1;
        self.stats.bytes_written += total + 24;
        Ok(DiskObjectHandle {
            planet_id: planet_id as u32,
            first_block: cursor,
            len: payload.len() as u64,
        })
    }

    /// Reads an object back and verifies its CRC before returning it.
    pub fn get_object(&mut self, handle: DiskObjectHandle) -> io::Result<Vec<u8>> {
        let planet_id = handle.planet_id as usize & (NUM_PLANETS - 1);
        let offset = block_offset(planet_id, handle.first_block);
        let mut header = [0u8; OBJECT_HEADER_BYTES as usize];
        self.file.read_exact_at(&mut header, offset)?;
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        let stored_crc = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let len = u64::from_le_bytes(header[8..16].try_into().unwrap());
        if magic != OBJECT_MAGIC {
            return Err(Error::new(ErrorKind::InvalidData, "no object at handle"));
        }
        if len != handle.len || len > DISK_PLANET_BYTES {
            return Err(Error::new(ErrorKind::InvalidData, "object length mismatch"));
        }
        let mut payload = vec![0u8; len as usize];
        self.file
            .read_exact_at(&mut payload, offset + OBJECT_HEADER_BYTES)?;
        if crc32(&payload) != stored_crc {
            return Err(Error::new(ErrorKind::InvalidData, "object CRC mismatch"));
        }
        self.stats.bytes_read += OBJECT_HEADER_BYTES + len;
        Ok(payload)
    }

    /// Tears down a team's disk room: punches the file hole back to the OS
    /// (physical reclaim on Linux) and resets the planet to empty.
    pub fn release_planet(&mut self, planet_id: usize) -> io::Result<u64> {
        let planet_id = planet_id & (NUM_PLANETS - 1);
        let used_blocks = match self.cursors[planet_id] {
            0 => 0,
            c => c,
        };
        punch_hole(&self.file, planet_base(planet_id), DISK_PLANET_BYTES)?;
        self.write_superblock(planet_id, FIRST_DATA_BLOCK)?;
        self.cursors[planet_id] = FIRST_DATA_BLOCK;
        self.stats.planets_released += 1;
        Ok(used_blocks.saturating_sub(FIRST_DATA_BLOCK) * DISK_BLOCK_BYTES)
    }

    /// Flushes everything a team produced down to the physical disk.
    pub fn sync(&self) -> io::Result<()> {
        self.file.sync_all()
    }

    /// Real bytes the host filesystem has committed for the sparse file.
    pub fn committed_bytes(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.blocks() * 512)
    }

    pub const fn virtual_bytes(&self) -> u64 {
        DISK_VIRTUAL_BYTES
    }

    pub fn stats(&self) -> DiskStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("cadran_disk_test_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn object_roundtrip() {
        let path = temp_path("roundtrip");
        let mut disk = CadranDiskManager::open(&path).unwrap();
        let payload: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let h = disk.put_object(7, &payload).unwrap();
        assert_eq!(disk.get_object(h).unwrap(), payload);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn objects_persist_across_reopen() {
        let path = temp_path("persist");
        let h;
        {
            let mut disk = CadranDiskManager::open(&path).unwrap();
            h = disk.put_object(3, b"the perpetual researcher paper #1").unwrap();
            disk.sync().unwrap();
        }
        let mut disk = CadranDiskManager::open(&path).unwrap();
        assert_eq!(
            disk.get_object(h).unwrap(),
            b"the perpetual researcher paper #1"
        );
        let h2 = disk.put_object(3, b"paper #2").unwrap();
        assert!(h2.first_block > h.first_block);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn corruption_is_detected() {
        let path = temp_path("corrupt");
        let mut disk = CadranDiskManager::open(&path).unwrap();
        let h = disk.put_object(1, b"important findings").unwrap();
        let mut byte = [0u8; 1];
        let off = block_offset(1, h.first_block) + OBJECT_HEADER_BYTES + 2;
        disk.file.read_exact_at(&mut byte, off).unwrap();
        byte[0] ^= 0xFF;
        disk.file.write_all_at(&byte, off).unwrap();
        assert!(disk.get_object(h).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn release_resets_planet() {
        let path = temp_path("release");
        let mut disk = CadranDiskManager::open(&path).unwrap();
        let h = disk.put_object(5, &vec![7u8; 50_000]).unwrap();
        let freed = disk.release_planet(5).unwrap();
        assert!(freed >= 50_000);
        assert!(disk.get_object(h).is_err());
        let h2 = disk.put_object(5, b"fresh start").unwrap();
        assert_eq!(h2.first_block, FIRST_DATA_BLOCK);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn planet_full_is_an_error_not_an_overflow() {
        let path = temp_path("full");
        let mut disk = CadranDiskManager::open(&path).unwrap();
        let big = vec![1u8; (DISK_PLANET_BYTES / 2) as usize];
        disk.put_object(9, &big).unwrap();
        disk.put_object(9, &big).unwrap_err();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn out_of_boundary_block_write_is_refused() {
        let path = temp_path("refused");
        let mut disk = CadranDiskManager::open(&path).unwrap();
        assert!(disk.write_block(2, DISK_BLOCKS_PER_PLANET, b"escape").is_err());
        assert!(disk.write_block(NUM_PLANETS, 0, b"escape").is_err());
        let mut buf = [0u8; 6];
        assert!(disk.read_block(2, DISK_BLOCKS_PER_PLANET + 5, &mut buf).is_err());
        disk.write_block(2, DISK_BLOCKS_PER_PLANET - 1, b"legal").unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
