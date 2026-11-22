//! Anvil region reading, export to a new region, and in-place chunk writes
//! as vanilla's `RegionFile` makes them (a file opened with `open_writable`).
//! Exports never overwrite a file: use a separate directory for offline
//! edits and copied .mcc data.
use horizon_core::position::ChunkPos;
use std::{
    cell::RefCell,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU32, Ordering},
};
use {
    crate::compression::{Compression, invalid},
    horizon_core::{
        nbt,
        nbt::{Limits, NamedTag},
    },
};

const SECTOR: usize = 4096;
const HEADER: usize = SECTOR * 2;
const MAX_COMPRESSED: usize = 64 * 1024 * 1024;

thread_local! {
    /// Decompression storage retained by each region-reading worker.
    static NBT_DECODE_BUFFER: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Debug)]
pub struct ChunkRecord {
    pub compression: Compression,
    pub compressed: Vec<u8>,
    pub timestamp: u32,
}
impl ChunkRecord {
    pub fn from_nbt(
        root: &NamedTag,
        compression: Compression,
        timestamp: u32,
        limits: Limits,
    ) -> io::Result<Self> {
        let bytes = nbt::write_named(root, limits).map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            compression,
            compressed: compression.encode(&bytes)?,
            timestamp,
        })
    }
    /// `RegionFileStorage.read`: the root read straight off the decompressing
    /// stream, so bytes after it, and a damaged stream past it, are never read.
    pub fn nbt(&self, limits: Limits) -> io::Result<NamedTag> {
        NBT_DECODE_BUFFER.with_borrow_mut(|bytes| {
            bytes.clear();
            let decoded = self
                .compression
                .decode_into(&self.compressed, limits.max_bytes, bytes);
            match nbt::read_named(bytes, limits) {
                Ok((root, _)) => Ok(root),
                Err(e) => Err(decoded.err().unwrap_or_else(|| invalid(e.to_string()))),
            }
        })
    }
}

/// What a region file's bytes are kept in: the file, or in tests a disk that
/// loses what was not synced when the power goes (`tests::sim`).
enum Disk {
    File(File),
    #[cfg(test)]
    Sim(tests::sim::Handle),
}
impl Disk {
    fn sync_data(&self) -> io::Result<()> {
        match self {
            Self::File(file) => file.sync_data(),
            #[cfg(test)]
            Self::Sim(sim) => sim.sync_data(),
        }
    }
}
impl Read for Disk {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::File(file) => file.read(buf),
            #[cfg(test)]
            Self::Sim(sim) => sim.read(buf),
        }
    }
}
impl Write for Disk {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::File(file) => file.write(buf),
            #[cfg(test)]
            Self::Sim(sim) => sim.write(buf),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::File(file) => file.flush(),
            #[cfg(test)]
            Self::Sim(_) => Ok(()),
        }
    }
}
impl Seek for Disk {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            Self::File(file) => file.seek(pos),
            #[cfg(test)]
            Self::Sim(sim) => sim.seek(pos),
        }
    }
}

pub struct RegionFile {
    file: Disk,
    path: PathBuf,
    position: ChunkPos,
    offsets: [u32; 1024],
    timestamps: [u32; 1024],
    /// The sectors in use (`RegionBitmap`): the header's two and every
    /// chunk's, and the old sectors of the chunks written or cleared since
    /// the last `sync` (`pending_free`).
    used: Vec<bool>,
    /// The first free sector (`RegionBitmap`'s low watermark): allocation
    /// starts its first-fit search here instead of at sector 0.
    first_free: usize,
    /// Whether `offsets` and `timestamps` differ from the header in the file:
    /// `write` and `clear` change them in memory, `sync` writes them.
    header_staged: bool,
    /// Whether chunk sectors were written since the last `sync`.
    sectors_written: bool,
    /// Whether a `c.X.Z.mcc` was renamed into the folder since the last `sync`.
    externals_written: bool,
    /// The sectors of the chunks written or cleared since the last `sync`
    /// (start, count). The header in the file may still point to them, so
    /// they are freed only once the new header is on the disk.
    pending_free: Vec<(usize, usize)>,
    /// The slots whose `c.X.Z.mcc` goes once the new header is on the disk:
    /// written inline or cleared since the last `sync`.
    pending_removals: Vec<usize>,
}

pub fn position_from_path(path: &Path) -> io::Result<ChunkPos> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| invalid("invalid region filename"))?;
    let parts: Vec<_> = name.split('.').collect();
    if parts.len() != 4 || parts[0] != "r" || parts[3] != "mca" {
        return Err(invalid("expected r.<x>.<z>.mca"));
    }
    let x: i32 = parts[1]
        .parse()
        .map_err(|_| invalid("invalid region x coordinate"))?;
    let z: i32 = parts[2]
        .parse()
        .map_err(|_| invalid("invalid region z coordinate"))?;
    // Regions outside this range cannot be addressed with signed chunk coordinates.
    if !(-67_108_864..=67_108_863).contains(&x) || !(-67_108_864..=67_108_863).contains(&z) {
        return Err(invalid("region coordinate outside chunk coordinate range"));
    }
    Ok(ChunkPos::new(x, z))
}

fn slot(region: ChunkPos, chunk: ChunkPos) -> io::Result<usize> {
    if chunk.region() != region {
        return Err(invalid("chunk does not belong to this region"));
    }
    let (x, z) = chunk.region_local();
    Ok(usize::from(x) + usize::from(z) * 32)
}
fn position(region: ChunkPos, index: usize) -> ChunkPos {
    ChunkPos::new(
        region.x * 32 + (index % 32) as i32,
        region.z * 32 + (index / 32) as i32,
    )
}
fn external_path(directory: &Path, chunk: ChunkPos) -> PathBuf {
    directory.join(format!("c.{}.{}.mcc", chunk.x, chunk.z))
}

fn write_region_header(
    file: &mut (impl Write + Seek),
    offsets: &[u32; 1024],
    timestamps: &[u32; 1024],
) -> io::Result<()> {
    let mut header = [0; HEADER];
    for i in 0..1024 {
        header[i * 4..i * 4 + 4].copy_from_slice(&offsets[i].to_be_bytes());
        header[SECTOR + i * 4..SECTOR + i * 4 + 4].copy_from_slice(&timestamps[i].to_be_bytes());
    }
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header)
}

/// The header of a region file, from its start: each chunk's place (sector
/// and sector count) and timestamp.
fn read_header(file: &mut (impl Read + Seek)) -> io::Result<([u32; 1024], [u32; 1024])> {
    let len = file.seek(SeekFrom::End(0))?;
    file.seek(SeekFrom::Start(0))?;
    // RegionFile's constructor: a short header (an empty or truncated file)
    // reads as far as it goes, the rest stays zero (no chunks).
    let mut header = [0; HEADER];
    let mut read = 0;
    while read < HEADER {
        match file.read(&mut header[read..])? {
            0 => break,
            n => read += n,
        }
    }
    let mut offsets = [0; 1024];
    let mut timestamps = [0; 1024];
    for i in 0..1024 {
        offsets[i] = u32::from_be_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
        timestamps[i] = u32::from_be_bytes(
            header[SECTOR + i * 4..SECTOR + i * 4 + 4]
                .try_into()
                .unwrap(),
        );
        if offsets[i] == 0 {
            continue;
        }
        let start = offsets[i] >> 8;
        let count = offsets[i] & 255;
        // Vanilla drops entries that overlap the header, are empty or start past
        // the end of the file (logged as invalid sectors).
        if start < 2 || count == 0 || u64::from(start) * SECTOR as u64 > len {
            offsets[i] = 0;
        }
    }
    Ok((offsets, timestamps))
}
/// `RegionFile.getChunkDataInputStream` for the chunk the header puts at
/// `offset`: a chunk the file cannot deliver is absent (`null`), not an
/// error, so a reader skips it the way vanilla does instead of giving up on
/// the whole region.
fn read_record(
    file: &mut (impl Read + Seek),
    directory: &Path,
    chunk: ChunkPos,
    offset: u32,
    timestamp: u32,
) -> io::Result<Option<ChunkRecord>> {
    if offset == 0 {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(u64::from(offset >> 8) * SECTOR as u64))?;
    let mut header = [0; 5];
    if file.read(&mut header)? < header.len() {
        return Ok(None); // "Chunk header is truncated"
    }
    let len = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
    if len == 0 {
        return Ok(None); // "Chunk is allocated, but stream is missing"
    }
    if len as u64 + 4 > (u64::from(offset & 255) * SECTOR as u64) {
        return Ok(None); // "Chunk stream is truncated"
    }
    let compression = Compression::try_from(header[4] & 127)?;
    let compressed = if header[4] & 128 != 0 {
        // `createExternalChunkInputStream`: the stub's length says nothing
        // about the stream in the `.mcc`, so a stub that also claims inline
        // data only warns; and a stub whose file is gone is a chunk that is
        // not there (`null`), not a damaged one.
        let path = external_path(directory, chunk);
        if !path.is_file() {
            return Ok(None);
        }
        let file = File::open(path)?;
        let mut data = Vec::new();
        file.take((MAX_COMPRESSED + 1) as u64)
            .read_to_end(&mut data)?;
        if data.len() > MAX_COMPRESSED {
            return Err(invalid("external chunk compressed limit exceeded"));
        }
        data
    } else {
        let mut data = vec![0; len - 1];
        match file.read_exact(&mut data) {
            Ok(()) => data,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
    };
    Ok(Some(ChunkRecord {
        compression,
        compressed,
        timestamp,
    }))
}
fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

impl RegionFile {
    pub fn position(&self) -> ChunkPos {
        self.position
    }
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_owned();
        let file = File::open(&path)?;
        Self::from_file(path, Disk::File(file))
    }

    /// Open for reading and in-place writes (`write`, `clear`, made durable
    /// by `sync`), creating an empty region with `create`.
    pub fn open_writable(path: impl AsRef<Path>, create: bool) -> io::Result<Self> {
        let path = path.as_ref().to_owned();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(create)
            .truncate(false)
            .open(&path)?;
        Self::from_file(path, Disk::File(file))
    }

    fn from_file(path: PathBuf, mut file: Disk) -> io::Result<Self> {
        let position = position_from_path(&path)?;
        let (offsets, timestamps) = read_header(&mut file)?;
        let mut region = Self {
            file,
            path,
            position,
            offsets,
            timestamps,
            used: Vec::new(),
            first_free: 0,
            header_staged: false,
            sectors_written: false,
            externals_written: false,
            pending_free: Vec::new(),
            pending_removals: Vec::new(),
        };
        region.force(0, 2);
        for offset in offsets.into_iter().filter(|&o| o != 0) {
            region.force((offset >> 8) as usize, (offset & 255) as usize);
        }
        region.first_free = region
            .used
            .iter()
            .position(|&u| !u)
            .unwrap_or(region.used.len());
        Ok(region)
    }

    /// `RegionBitmap.force`.
    fn force(&mut self, start: usize, count: usize) {
        if self.used.len() < start + count {
            self.used.resize(start + count, false);
        }
        self.used[start..start + count].fill(true);
        if self.first_free >= start && self.first_free < start + count {
            self.first_free = self.used[start + count..]
                .iter()
                .position(|&u| !u)
                .map_or(self.used.len(), |i| start + count + i);
        }
    }
    /// `RegionBitmap.free`.
    fn free(&mut self, start: usize, count: usize) {
        let end = (start + count).min(self.used.len());
        if start < end {
            self.used[start..end].fill(false);
            self.first_free = self.first_free.min(start);
        }
    }
    /// `RegionBitmap.allocate`: the first run of free sectors long enough
    /// (the end of the file is free).
    fn allocate(&mut self, count: usize) -> usize {
        let mut current = self.first_free;
        loop {
            let start = (current..self.used.len())
                .find(|&s| !self.used[s])
                .unwrap_or(self.used.len());
            let end = (start..self.used.len()).find(|&s| self.used[s]);
            if end.is_none_or(|end| end - start >= count) {
                self.force(start, count);
                return start;
            }
            current = end.unwrap_or(start);
        }
    }

    /// `RegionFile.write`: the chunk goes to the first free run of sectors
    /// that holds it (an oversized one, 256 sectors or more, to its external
    /// `c.X.Z.mcc` file with a one-sector stub), the header points to it,
    /// and only then are its old sectors freed. The other chunks stay where
    /// they are. A chunk's last sector is padded with zeros.
    ///
    /// The sectors are written now and the header in memory points to them,
    /// so this handle reads the new chunk at once; the file's header follows
    /// at the next [`sync`](Self::sync) (or when the handle is dropped),
    /// which frees the old sectors.
    pub fn write(&mut self, chunk: ChunkPos, record: &ChunkRecord) -> io::Result<()> {
        let i = slot(self.position, chunk)?;
        if record.compressed.len() > MAX_COMPRESSED {
            return Err(invalid("compressed chunk limit exceeded"));
        }
        let required = (record.compressed.len() + 5).div_ceil(SECTOR);
        let external = required >= 256;
        let sectors = if external { 1 } else { required };
        let start = self.allocate(sectors);
        if start + sectors > 0x100_0000 {
            self.free(start, sectors);
            return Err(invalid("region sector address limit exceeded"));
        }
        let directory = self.path.parent().unwrap_or(Path::new(".")).to_owned();
        let external_file = external_path(&directory, chunk);
        let mut data = Vec::with_capacity(sectors * SECTOR);
        if external {
            let temporary = directory.join(format!(".c.{}.{}.mcc.tmp", chunk.x, chunk.z));
            let written = File::create(&temporary)
                .and_then(|mut f| {
                    f.write_all(&record.compressed)?;
                    // The region's sync only covers the stub, not this external payload.
                    f.sync_all()
                })
                .and_then(|_| fs::rename(&temporary, &external_file));
            if let Err(e) = written {
                let _ = fs::remove_file(&temporary);
                self.free(start, sectors);
                return Err(e);
            }
            data.extend_from_slice(&1u32.to_be_bytes());
            data.push(record.compression as u8 | 128);
        } else {
            data.extend_from_slice(&(record.compressed.len() as u32 + 1).to_be_bytes());
            data.push(record.compression as u8);
            data.extend_from_slice(&record.compressed);
        }
        data.resize(sectors * SECTOR, 0);
        let written = self
            .file
            .seek(SeekFrom::Start((start * SECTOR) as u64))
            .and_then(|_| self.file.write_all(&data));
        if let Err(e) = written {
            self.free(start, sectors);
            return Err(e);
        }
        self.sectors_written = true;
        if external {
            // The `.mcc` just written is this chunk's: the one an earlier
            // write or clear left to remove is gone already, replaced.
            self.pending_removals.retain(|&s| s != i);
            self.externals_written = true;
        } else if !self.pending_removals.contains(&i) {
            self.pending_removals.push(i);
        }
        let old = self.offsets[i];
        self.offsets[i] = (start as u32) << 8 | sectors as u32;
        self.timestamps[i] = record.timestamp;
        self.stage_old_sectors(old);
        Ok(())
    }

    /// `RegionFile.clear`: the chunk is gone; `timestamp` marks when.
    /// Returns whether the region held it. As with `write`, the file's header
    /// follows at the next `sync`, which frees the sectors and removes the
    /// chunk's `.mcc`.
    pub fn clear(&mut self, chunk: ChunkPos, timestamp: u32) -> io::Result<bool> {
        let i = slot(self.position, chunk)?;
        let old = self.offsets[i];
        if old == 0 {
            return Ok(false);
        }
        self.offsets[i] = 0;
        self.timestamps[i] = timestamp;
        if !self.pending_removals.contains(&i) {
            self.pending_removals.push(i);
        }
        self.stage_old_sectors(old);
        Ok(true)
    }

    /// The header in memory changed; the sectors at `old` stay taken until
    /// the file's header no longer points to them (`sync`).
    fn stage_old_sectors(&mut self, old: u32) {
        self.header_staged = true;
        if old != 0 {
            self.pending_free
                .push(((old >> 8) as usize, (old & 255) as usize));
        }
    }

    /// Make the writes and clears since the last `sync` durable, in an order
    /// that leaves every chunk the file's header names readable whatever a
    /// power cut keeps of what was not synced: the chunks' new sectors (and
    /// the folder their new `.mcc` files were renamed into) reach the disk,
    /// then the header that points to them; only once that header is on the
    /// disk are the old sectors free for the next write and the `.mcc` files
    /// no chunk uses any more removed.
    ///
    /// Vanilla gets this order by opening the file `DSYNC`: every chunk
    /// write and header write is on the disk before the next starts. Here a
    /// batch of writes costs two syncs (one when it only clears) and one
    /// header write, not two of each a chunk.
    pub fn sync(&mut self) -> io::Result<()> {
        if !self.header_staged {
            return self.file.sync_data();
        }
        if self.sectors_written {
            self.file.sync_data()?;
            self.sectors_written = false;
        }
        let directory = self.path.parent().unwrap_or(Path::new(".")).to_owned();
        if self.externals_written {
            crate::fs_sync::sync_dir(&directory)?;
            self.externals_written = false;
        }
        self.write_header()?;
        self.file.sync_data()?;
        self.header_staged = false;
        for (start, count) in std::mem::take(&mut self.pending_free) {
            self.free(start, count);
        }
        for i in std::mem::take(&mut self.pending_removals) {
            remove_if_exists(&external_path(&directory, position(self.position, i)))?;
        }
        Ok(())
    }

    /// `RegionFile.writeHeader`: both header sectors from memory, so the
    /// entries dropped as invalid when the file was opened are zero on disk too.
    fn write_header(&mut self) -> io::Result<()> {
        write_region_header(&mut self.file, &self.offsets, &self.timestamps)
    }

    pub fn chunks(&self) -> impl Iterator<Item = (ChunkPos, u32)> + '_ {
        self.offsets
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(i, _)| (position(self.position, i), self.timestamps[i]))
    }

    /// The chunk's record, `None` for a chunk the file does not hold or cannot
    /// deliver (`read_record`).
    pub fn read(&mut self, chunk: ChunkPos) -> io::Result<Option<ChunkRecord>> {
        let i = slot(self.position, chunk)?;
        let directory = self.path.parent().unwrap_or(Path::new("."));
        read_record(
            &mut self.file,
            directory,
            chunk,
            self.offsets[i],
            self.timestamps[i],
        )
    }

    /// The chunk as [`read`](Self::read) would give it with its `c.X.Z.mcc`
    /// taken from `directory`: `None` unless its stub marks it as stored in
    /// that file (and that file is there).
    pub fn read_external_from(
        &mut self,
        chunk: ChunkPos,
        directory: &Path,
    ) -> io::Result<Option<ChunkRecord>> {
        if !self.is_external(chunk)? {
            return Ok(None);
        }
        let i = slot(self.position, chunk)?;
        read_record(
            &mut self.file,
            directory,
            chunk,
            self.offsets[i],
            self.timestamps[i],
        )
    }

    /// Whether the chunk's stub marks it as stored in its own `c.X.Z.mcc`.
    pub fn is_external(&mut self, chunk: ChunkPos) -> io::Result<bool> {
        let offset = self.offsets[slot(self.position, chunk)?];
        if offset == 0 {
            return Ok(false);
        }
        self.file
            .seek(SeekFrom::Start(u64::from(offset >> 8) * SECTOR as u64))?;
        let mut header = [0; 5];
        Ok(self.file.read(&mut header)? == header.len() && header[4] & 128 != 0)
    }

    /// The `c.X.Z.mcc` files this region's oversized chunks are stored in
    /// (`createExternalChunkPath`): one per chunk whose stub marks it as
    /// external ([`is_external`](Self::is_external)), in the folder this
    /// region was opened from. Empty when none of its chunks is oversized.
    pub fn external_payloads(&mut self) -> io::Result<Vec<PathBuf>> {
        let directory = self.path.parent().unwrap_or(Path::new(".")).to_owned();
        let positions: Vec<ChunkPos> = self.chunks().map(|(pos, _)| pos).collect();
        let mut out = Vec::new();
        for pos in positions {
            if self.is_external(pos)? {
                out.push(external_path(&directory, pos));
            }
        }
        Ok(out)
    }

    /// Export all chunks, preserving untouched compressed bytes and timestamps.
    /// Overrides are keyed by chunk position; None deletes a chunk in the export.
    pub fn export(
        &mut self,
        destination: impl AsRef<Path>,
        changes: &[(ChunkPos, Option<ChunkRecord>)],
    ) -> io::Result<()> {
        let destination = destination.as_ref();
        if position_from_path(destination)? != self.position {
            return Err(invalid("export region coordinates differ"));
        }
        let mut replacements = BTreeChanges::new();
        for (chunk, record) in changes {
            let i = slot(self.position, *chunk)?;
            if replacements.insert(i, record.as_ref()).is_some() {
                return Err(invalid("duplicate chunk override"));
            }
        }
        let mut writer = RegionWriter::create(destination)?;
        for i in 0..1024 {
            let chunk = position(self.position, i);
            if let Some(record) = replacements.get(&i) {
                if let Some(record) = record {
                    writer.write(chunk, record)?;
                }
            } else if let Some(record) = self.read(chunk)? {
                writer.write(chunk, &record)?;
            }
        }
        writer.finish()
    }
}
impl Drop for RegionFile {
    /// A handle closed with writes not yet synced makes them durable as
    /// `sync` does, so the file's header names them for the next reader. An
    /// error here has no caller to go to; a writer that must know syncs.
    fn drop(&mut self) {
        if self.header_staged {
            let _ = self.sync();
        }
    }
}
type BTreeChanges<'a> = std::collections::BTreeMap<usize, Option<&'a ChunkRecord>>;

/// A region file's header kept without the file open: which chunks the file
/// holds and where, as of when it was read. For a reader of many chunks that
/// must not keep every file open (far terrain: a region file for each 32
/// chunks of the view), so that the header is read once and not once a chunk.
///
/// It goes stale when the file is written: `RegionFile::write` moves the
/// chunk, and its `sync` frees the old sectors for the next chunk written.
/// Whoever keeps a header must know when that happens.
pub struct RegionHeader {
    path: PathBuf,
    position: ChunkPos,
    offsets: [u32; 1024],
    timestamps: [u32; 1024],
}
impl RegionHeader {
    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_owned();
        let position = position_from_path(&path)?;
        let (offsets, timestamps) = read_header(&mut File::open(&path)?)?;
        Ok(Self {
            path,
            position,
            offsets,
            timestamps,
        })
    }
    /// `RegionFile::read` at the place this header gives the chunk, with the
    /// file opened for this one record.
    pub fn read_chunk(&self, chunk: ChunkPos) -> io::Result<Option<ChunkRecord>> {
        let i = slot(self.position, chunk)?;
        if self.offsets[i] == 0 {
            return Ok(None);
        }
        let directory = self.path.parent().unwrap_or(Path::new("."));
        read_record(
            &mut File::open(&self.path)?,
            directory,
            chunk,
            self.offsets[i],
            self.timestamps[i],
        )
    }
}

static NEXT_TEMP: AtomicU32 = AtomicU32::new(0);

/// Writes a complete new region to a temporary sibling, then publishes it with
/// an atomic, non-overwriting hard link. Drop rolls back unpublished files.
pub struct RegionWriter {
    file: File,
    temporary: PathBuf,
    destination: PathBuf,
    position: ChunkPos,
    offsets: [u32; 1024],
    timestamps: [u32; 1024],
    next_sector: u32,
    externals: Vec<PathBuf>,
    committed: bool,
    failed: bool,
}
impl RegionWriter {
    pub fn create(destination: impl AsRef<Path>) -> io::Result<Self> {
        let destination = destination.as_ref().to_owned();
        let position = position_from_path(&destination)?;
        if destination.try_exists()? {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "destination region already exists",
            ));
        }
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let (temporary, file) = loop {
            let path = parent.join(format!(
                ".horizon-{}-{}.tmp",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => break (path, file),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        };
        let mut writer = Self {
            file,
            temporary,
            destination,
            position,
            offsets: [0; 1024],
            timestamps: [0; 1024],
            next_sector: 2,
            externals: Vec::new(),
            committed: false,
            failed: false,
        };
        writer.file.write_all(&[0; HEADER])?;
        Ok(writer)
    }

    pub fn write(&mut self, chunk: ChunkPos, record: &ChunkRecord) -> io::Result<()> {
        if self.failed {
            return Err(invalid("region writer failed previously"));
        }
        let result = self.write_record(chunk, record);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn write_record(&mut self, chunk: ChunkPos, record: &ChunkRecord) -> io::Result<()> {
        let i = slot(self.position, chunk)?;
        if self.offsets[i] != 0 {
            return Err(invalid("chunk already written"));
        }
        if record.compressed.len() > MAX_COMPRESSED {
            return Err(invalid("compressed chunk limit exceeded"));
        }
        let required = (record.compressed.len() + 5).div_ceil(SECTOR);
        let external = required >= 256;
        let sectors = if external { 1 } else { required as u32 };
        if self.next_sector + sectors > 0x100_0000 {
            return Err(invalid("region sector address limit exceeded"));
        }
        if external {
            let path = external_path(self.destination.parent().unwrap_or(Path::new(".")), chunk);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            self.externals.push(path);
            file.write_all(&record.compressed)?;
            file.sync_all()?;
        }
        self.file
            .seek(SeekFrom::Start(u64::from(self.next_sector) * SECTOR as u64))?;
        let len = if external {
            1
        } else {
            record.compressed.len() as u32 + 1
        };
        let [a, b, c, d] = len.to_be_bytes();
        let compression = record.compression as u8 | if external { 128 } else { 0 };
        self.file.write_all(&[a, b, c, d, compression])?;
        if !external {
            self.file.write_all(&record.compressed)?;
        }
        self.offsets[i] = self.next_sector << 8 | sectors;
        self.timestamps[i] = record.timestamp;
        self.next_sector += sectors;
        Ok(())
    }

    /// The `c.X.Z.mcc` files written so far beside the destination, one per
    /// oversized chunk.
    pub fn external_files(&self) -> &[PathBuf] {
        &self.externals
    }

    pub fn finish(mut self) -> io::Result<()> {
        if self.failed {
            return Err(invalid("cannot publish a failed region writer"));
        }
        self.file
            .set_len(u64::from(self.next_sector) * SECTOR as u64)?;
        write_region_header(&mut self.file, &self.offsets, &self.timestamps)?;
        self.file.sync_all()?;
        // Hard-link publication fails atomically if another writer won the name.
        fs::hard_link(&self.temporary, &self.destination)?;
        self.committed = true;
        Ok(())
    }
}
impl Drop for RegionWriter {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.temporary);
        if !self.committed {
            for path in &self.externals {
                let _ = fs::remove_file(path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    /// The pre-hint allocator: first fit from sector 0. The hint must keep
    /// producing exactly its placements.
    fn naive_first_fit(used: &[bool], count: usize) -> usize {
        let mut current = 0;
        loop {
            let start = (current..used.len())
                .find(|&s| !used[s])
                .unwrap_or(used.len());
            let end = (start..used.len()).find(|&s| used[s]);
            if end.is_none_or(|end| end - start >= count) {
                return start;
            }
            current = end.unwrap_or(start);
        }
    }

    #[test]
    fn allocation_matches_first_fit_from_zero() {
        let dir =
            std::env::temp_dir().join(format!("horizon-region-first-free-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.0.0.mca");
        let mut region = RegionFile::open_writable(&path, true).unwrap();

        let mut seed = 0x1234_5678u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for op in 0..120 {
            let slot = next() % 1024;
            let size = match next() % 4 {
                0 => 1, // external: one stub sector
                _ => 2,
            };
            let pos = position(region.position, slot);
            if next() % 5 == 0 {
                assert!(region.clear(pos, op as u32).is_ok());
            } else {
                let record = ChunkRecord {
                    compression: Compression::None,
                    compressed: vec![0; size * SECTOR - 5],
                    timestamp: op as u32,
                };
                assert!(region.write(pos, &record).is_ok());
            }
            // The old sectors are free once the header is synced.
            region.sync().unwrap();
            // The naive bitmap from what the region placed, under the old rule.
            let mut naive = vec![false; region.used.len()];
            naive[0..2].fill(true);
            for i in 0..1024 {
                let o = region.offsets[i];
                if o != 0 {
                    let s = (o >> 8) as usize;
                    let c = (o & 255) as usize;
                    if naive.len() < s + c {
                        naive.resize(s + c, false);
                    }
                    naive[s..s + c].fill(true);
                }
            }
            assert_eq!(region.used, naive);
            assert_eq!(region.first_free, naive_first_fit(&naive, 1));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_kept_header_reads_the_chunks_of_its_file() {
        let dir =
            std::env::temp_dir().join(format!("horizon-region-header-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.-1.2.mca");
        let record = |fill: u8, len: usize| ChunkRecord {
            compression: Compression::None,
            compressed: vec![fill; len],
            timestamp: u32::from(fill),
        };
        let (first, last) = (ChunkPos::new(-32, 64), ChunkPos::new(-1, 95));
        let missing = ChunkPos::new(-31, 64);
        let mut region = RegionFile::open_writable(&path, true).unwrap();
        region.write(first, &record(7, 100)).unwrap();
        // Two sectors, after the first chunk's one.
        region.write(last, &record(9, 5000)).unwrap();
        region.sync().unwrap();

        let header = RegionHeader::read(&path).unwrap();
        for (chunk, fill, len) in [(first, 7, 100), (last, 9, 5000)] {
            let read = header.read_chunk(chunk).unwrap().unwrap();
            assert_eq!(read.compressed, vec![fill; len]);
            assert_eq!(read.timestamp, u32::from(fill));
        }
        assert!(header.read_chunk(missing).unwrap().is_none());
        // The next region's chunk.
        assert!(header.read_chunk(ChunkPos::new(0, 64)).is_err());
        assert!(RegionHeader::read(dir.join("r.5.5.mca")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A chunk's `.mcc` stays while the header on the disk names its stub:
    /// written inline again it goes at the `sync`, and cleared then written
    /// oversized again in one batch it is the new one and stays.
    #[test]
    fn an_mcc_goes_only_once_the_synced_header_stops_naming_it() {
        let dir = std::env::temp_dir().join(format!("horizon-region-mcc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.0.0.mca");
        let (chunk, mcc) = (ChunkPos::new(3, 4), dir.join("c.3.4.mcc"));
        let record = |fill: u8, len: usize| ChunkRecord {
            compression: Compression::None,
            compressed: vec![fill; len],
            timestamp: u32::from(fill),
        };
        let (big, small) = (record(7, 256 * SECTOR), record(8, 100));
        let read = || {
            RegionFile::open(&path)
                .unwrap()
                .read(chunk)
                .unwrap()
                .unwrap()
        };
        let mut region = RegionFile::open_writable(&path, true).unwrap();
        region.write(chunk, &big).unwrap();
        region.sync().unwrap();
        assert!(mcc.exists());

        region.write(chunk, &small).unwrap();
        assert!(mcc.exists());
        assert_eq!(read().compressed, big.compressed);
        region.sync().unwrap();
        assert!(!mcc.exists());
        assert_eq!(read().compressed, small.compressed);

        assert!(region.clear(chunk, 9).unwrap());
        region.write(chunk, &big).unwrap();
        region.sync().unwrap();
        assert!(mcc.exists());
        assert_eq!(read().compressed, big.compressed);
        drop(region);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A disk with a volatile write cache, for power cuts. What was synced
    /// stays. Of each 512-byte block written since the last sync, a cut keeps
    /// any one of its versions: none, the last, or one in between, each block
    /// chosen on its own. That covers the kernel writing a file's dirty pages
    /// back in any order, and a drive tearing a write between its sectors.
    pub(super) mod sim {
        use std::{
            cell::RefCell,
            collections::BTreeMap,
            io::{self, SeekFrom},
            rc::Rc,
        };

        const BLOCK: usize = 512;

        #[derive(Default)]
        pub(crate) struct Drive {
            /// The bytes as the program reads them.
            cache: Vec<u8>,
            /// The bytes as of the last sync.
            durable: Vec<u8>,
            /// Each block written since the last sync: its bytes after each
            /// of those writes, oldest first.
            versions: BTreeMap<usize, Vec<Vec<u8>>>,
            /// Writes and syncs so far.
            pub(crate) ops: usize,
            /// The cut: before the operation numbered `.0`, its choices
            /// drawn from the seed `.1`.
            cut: Option<(usize, u64)>,
            image: Option<Vec<u8>>,
        }
        impl Drive {
            pub(crate) fn new(cut: Option<(usize, u64)>) -> Self {
                Self {
                    cut,
                    ..Self::default()
                }
            }
            /// A drive holding `bytes`, all of them synced.
            pub(crate) fn synced(bytes: Vec<u8>) -> Self {
                Self {
                    durable: bytes.clone(),
                    cache: bytes,
                    ..Self::default()
                }
            }
            fn before_op(&mut self) {
                if let Some((at, seed)) = self.cut
                    && at == self.ops
                {
                    self.image = Some(self.crash(seed));
                }
                self.ops += 1;
            }
            /// What a power cut now leaves on the disk.
            pub(crate) fn crash(&self, seed: u64) -> Vec<u8> {
                let mut rng = super::Rng(seed);
                let mut image = self.durable.clone();
                // The file's new length reached the disk, or it did not.
                if rng.below(2) == 0 {
                    image.resize(self.cache.len(), 0);
                }
                for (&block, versions) in &self.versions {
                    let start = block * BLOCK;
                    if start >= image.len() {
                        continue;
                    }
                    let kept = match rng.below(versions.len() + 1) {
                        0 => continue,
                        n => &versions[n - 1],
                    };
                    let end = (start + BLOCK).min(image.len());
                    image[start..end].copy_from_slice(&kept[..end - start]);
                }
                image
            }
            /// The image of the cut, or of a cut after the last operation.
            pub(crate) fn take_image(&mut self, seed: u64) -> Vec<u8> {
                self.image.take().unwrap_or_else(|| self.crash(seed))
            }
        }

        /// A handle on a [`Drive`], with its own position.
        pub(crate) struct Handle {
            drive: Rc<RefCell<Drive>>,
            at: u64,
        }
        impl Handle {
            pub(crate) fn new(drive: Rc<RefCell<Drive>>) -> Self {
                Self { drive, at: 0 }
            }
            pub(crate) fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                let drive = self.drive.borrow();
                let start = (self.at as usize).min(drive.cache.len());
                let n = buf.len().min(drive.cache.len() - start);
                buf[..n].copy_from_slice(&drive.cache[start..start + n]);
                self.at += n as u64;
                Ok(n)
            }
            pub(crate) fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                let mut drive = self.drive.borrow_mut();
                drive.before_op();
                let (start, end) = (self.at as usize, self.at as usize + buf.len());
                if drive.cache.len() < end {
                    drive.cache.resize(end, 0);
                }
                drive.cache[start..end].copy_from_slice(buf);
                for block in start / BLOCK..end.div_ceil(BLOCK) {
                    let mut bytes = drive.cache
                        [block * BLOCK..((block + 1) * BLOCK).min(drive.cache.len())]
                        .to_vec();
                    bytes.resize(BLOCK, 0);
                    drive.versions.entry(block).or_default().push(bytes);
                }
                self.at = end as u64;
                Ok(buf.len())
            }
            pub(crate) fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
                let len = self.drive.borrow().cache.len() as i64;
                let at = match pos {
                    SeekFrom::Start(n) => n as i64,
                    SeekFrom::End(n) => len + n,
                    SeekFrom::Current(n) => self.at as i64 + n,
                };
                self.at = u64::try_from(at).map_err(|_| io::Error::other("seek before 0"))?;
                Ok(self.at)
            }
            pub(crate) fn sync_data(&self) -> io::Result<()> {
                let mut drive = self.drive.borrow_mut();
                drive.before_op();
                let drive = &mut *drive;
                drive.durable.resize(drive.cache.len(), 0);
                for (block, versions) in std::mem::take(&mut drive.versions) {
                    let start = block * BLOCK;
                    let end = (start + BLOCK).min(drive.durable.len());
                    if start < end {
                        drive.durable[start..end]
                            .copy_from_slice(&versions.last().unwrap()[..end - start]);
                    }
                }
                Ok(())
            }
        }
    }

    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            // splitmix64
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) % n as u64) as usize
        }
    }

    /// The chunks the power-cut runs write, in region 0 0.
    const SLOTS: usize = 40;
    const BATCHES: usize = 40;

    /// A record that names its chunk and version, and says so in every byte.
    fn tagged(slot: usize, version: u32, len: usize) -> Vec<u8> {
        let mut bytes = vec![0; len.max(6)];
        bytes[..2].copy_from_slice(&(slot as u16).to_be_bytes());
        bytes[2..6].copy_from_slice(&version.to_be_bytes());
        for (j, b) in bytes[6..].iter_mut().enumerate() {
            *b = (slot as u32 * 31 + version * 7 + j as u32) as u8;
        }
        bytes
    }
    /// The chunk and version of a [`tagged`] record, `None` for any other bytes.
    fn untag(bytes: &[u8]) -> Option<(usize, u32)> {
        let slot = usize::from(u16::from_be_bytes(bytes.get(..2)?.try_into().ok()?));
        let version = u32::from_be_bytes(bytes.get(2..6)?.try_into().ok()?);
        (tagged(slot, version, bytes.len()) == bytes).then_some((slot, version))
    }

    #[derive(Clone, Copy)]
    enum Order {
        /// `write`, `clear` and `sync`.
        Synced,
        /// The order before it: the header written after each chunk's sectors
        /// with no sync between, the old sectors free at once, one sync a batch.
        Unsynced,
    }
    fn write_unsynced(region: &mut RegionFile, chunk: ChunkPos, record: &ChunkRecord) {
        let i = slot(region.position, chunk).unwrap();
        let sectors = (record.compressed.len() + 5).div_ceil(SECTOR);
        let start = region.allocate(sectors);
        let mut data = (record.compressed.len() as u32 + 1).to_be_bytes().to_vec();
        data.push(record.compression as u8);
        data.extend_from_slice(&record.compressed);
        data.resize(sectors * SECTOR, 0);
        region
            .file
            .seek(SeekFrom::Start((start * SECTOR) as u64))
            .unwrap();
        region.file.write_all(&data).unwrap();
        let old = region.offsets[i];
        region.offsets[i] = (start as u32) << 8 | sectors as u32;
        region.timestamps[i] = record.timestamp;
        region.write_header().unwrap();
        if old != 0 {
            region.free((old >> 8) as usize, (old & 255) as usize);
        }
    }
    fn clear_unsynced(region: &mut RegionFile, chunk: ChunkPos, timestamp: u32) {
        let i = slot(region.position, chunk).unwrap();
        let old = region.offsets[i];
        if old != 0 {
            region.offsets[i] = 0;
            region.timestamps[i] = timestamp;
            region.write_header().unwrap();
            region.free((old >> 8) as usize, (old & 255) as usize);
        }
    }

    /// A batch of a power-cut run: the drive's operations `start..end`, the
    /// acknowledged versions before it, and what it wrote (`None` clears).
    struct Batch {
        start: usize,
        end: usize,
        before: Vec<Option<u32>>,
        writes: Vec<(usize, Option<u32>)>,
        after: Vec<Option<u32>>,
    }

    /// `BATCHES` batches of writes and clears of one to four sectors each,
    /// each batch acknowledged once its sync returns, on a drive cut at
    /// `cut`. The batches, and what the cut left on the disk.
    fn power_cut_run(order: Order, seed: u64, cut: Option<(usize, u64)>) -> (Vec<Batch>, Vec<u8>) {
        let drive = Rc::new(RefCell::new(sim::Drive::new(cut)));
        let disk = Disk::Sim(sim::Handle::new(drive.clone()));
        let mut region = RegionFile::from_file(PathBuf::from("r.0.0.mca"), disk).unwrap();
        let mut rng = Rng(seed);
        let mut acked = vec![None; SLOTS];
        let mut version = 0;
        let mut batches = Vec::new();
        for _ in 0..BATCHES {
            let start = drive.borrow().ops;
            let before = acked.clone();
            let mut writes = Vec::new();
            for _ in 0..1 + rng.below(10) {
                let s = rng.below(SLOTS);
                let chunk = position(region.position, s);
                version += 1;
                if rng.below(6) == 0 {
                    match order {
                        Order::Synced => {
                            region.clear(chunk, version).unwrap();
                        }
                        Order::Unsynced => clear_unsynced(&mut region, chunk, version),
                    }
                    writes.push((s, None));
                } else {
                    let record = ChunkRecord {
                        compression: Compression::None,
                        compressed: tagged(s, version, 6 + rng.below(4 * SECTOR - 11)),
                        timestamp: version,
                    };
                    match order {
                        Order::Synced => region.write(chunk, &record).unwrap(),
                        Order::Unsynced => write_unsynced(&mut region, chunk, &record),
                    }
                    writes.push((s, Some(version)));
                }
            }
            match order {
                Order::Synced => region.sync().unwrap(),
                Order::Unsynced => region.file.sync_data().unwrap(),
            }
            for &(s, v) in &writes {
                acked[s] = v;
            }
            let end = drive.borrow().ops;
            let after = acked.clone();
            batches.push(Batch {
                start,
                end,
                before,
                writes,
                after,
            });
        }
        drop(region);
        let image = drive.borrow_mut().take_image(cut.map_or(seed, |c| c.1));
        (batches, image)
    }

    /// The acknowledged chunks a cut at `cut` lost (gone, or an older
    /// version), and the reads that gave another chunk's bytes or failed.
    fn after_the_cut(batches: &[Batch], image: Vec<u8>, cut: usize) -> (usize, usize) {
        let drive = Rc::new(RefCell::new(sim::Drive::synced(image)));
        let disk = Disk::Sim(sim::Handle::new(drive));
        let mut region =
            RegionFile::from_file(PathBuf::from("r.0.0.mca"), disk).expect("a crash image opens");
        let batch = batches.iter().find(|b| b.start <= cut && cut < b.end);
        let (mut lost, mut wrong) = (0, 0);
        for s in 0..SLOTS {
            // The acknowledged version, or one the batch in flight wrote.
            let allowed: Vec<Option<u32>> = match batch {
                Some(b) => std::iter::once(b.before[s])
                    .chain(b.writes.iter().filter(|w| w.0 == s).map(|w| w.1))
                    .collect(),
                None => vec![batches.last().unwrap().after[s]],
            };
            match region.read(position(region.position, s)) {
                Ok(None) if !allowed.contains(&None) => lost += 1,
                Ok(None) => {}
                Ok(Some(record)) => match untag(&record.compressed) {
                    Some((got, v)) if got == s => lost += usize::from(!allowed.contains(&Some(v))),
                    _ => wrong += 1,
                },
                Err(_) => wrong += 1,
            }
        }
        (lost, wrong)
    }

    /// Cuts at operations spread over `runs` workloads: the chunks lost and
    /// the wrong reads in all of them.
    fn power_cuts(order: Order, runs: u64, cuts: usize) -> (usize, usize) {
        let (mut lost, mut wrong) = (0, 0);
        for run in 0..runs {
            let seed = 0x5eed_0000 + run;
            let (batches, _) = power_cut_run(order, seed, None);
            let ops = batches.last().unwrap().end;
            let mut rng = Rng(seed ^ 0xc0ffee);
            for _ in 0..cuts {
                let cut = rng.below(ops + 1);
                let (batches, image) =
                    power_cut_run(order, seed, Some((cut, rng.below(usize::MAX) as u64)));
                let (l, w) = after_the_cut(&batches, image, cut);
                lost += l;
                wrong += w;
            }
        }
        (lost, wrong)
    }

    /// Whatever a power cut keeps of the writes since the last sync, every
    /// chunk a synced batch wrote reads back as that version or one the batch
    /// in flight wrote, and no read gives another chunk's bytes.
    #[test]
    fn a_power_cut_loses_no_synced_chunk_and_mixes_none_up() {
        assert_eq!(power_cuts(Order::Synced, 3, 200), (0, 0));
    }

    /// The test above can fail: the old order loses synced chunks and reads
    /// other chunks' bytes under the same cuts.
    #[test]
    fn the_old_write_order_loses_and_mixes_up_synced_chunks() {
        let (lost, wrong) = power_cuts(Order::Unsynced, 3, 200);
        assert!(lost > 0, "lost {lost}");
        assert!(wrong > 0, "wrong {wrong}");
    }
}
