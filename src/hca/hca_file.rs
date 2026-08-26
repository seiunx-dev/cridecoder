//! High-level HCA decoder with streaming capabilities

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};

pub use super::decoder::HcaInfo;
use super::decoder::{pcm_f32_to_i16, ClHca, HcaError, HCA_SUBFRAMES};
use super::imdct::imdct_overlap;

/// Key test parameters for testing HCA decryption keys
#[derive(Debug, Clone, Default)]
pub struct KeyTest {
    pub key: u64,
    pub subkey: u64,
    pub start_offset: u32,
    pub best_score: i32,
    pub best_key: u64,
}

// Key testing constants
const HCA_KEY_SCORE_SCALE: i32 = 10;
const HCA_KEY_MAX_SKIP_BLANKS: i32 = 1200;
const HCA_KEY_MIN_TEST_FRAMES: i32 = 3;
const HCA_KEY_MAX_TEST_FRAMES: i32 = 7;
const HCA_KEY_MAX_FRAME_SCORE: i32 = 600;
const HCA_KEY_MAX_TOTAL_SCORE: i32 = HCA_KEY_MAX_TEST_FRAMES * 50 * HCA_KEY_SCORE_SCALE;

/// High-level HCA decoder wrapping the low-level ClHCA decoder with streaming capabilities
pub struct HcaDecoder<R: Read + Seek> {
    reader: R,
    info: HcaInfo,
    // Boxed: ClHca is large; keeping it inline makes HcaDecoder big enough that a
    // few decoders in one stack frame overflow a 2 MB thread stack.
    handle: Box<ClHca>,
    buf: Vec<u8>,
    /// Multi-block read-ahead buffer: sequential decoding otherwise issues one
    /// tiny read syscall per block (~4% of decode time on macOS).
    chunk: Vec<u8>,
    /// First block held in `chunk`; `u32::MAX` marks the buffer invalid.
    chunk_first: u32,
    /// Number of blocks held in `chunk`.
    chunk_blocks: u32,
    fbuf: Vec<f32>,
    current_delay: i32,
    current_block: u32,
    reader_offset: Option<u64>,
    owns_file: bool,
    /// Valid output sample frames emitted so far (after front-delay trim), used to
    /// drop the trailing encoder_padding so total output matches CRI's count.
    samples_written: u64,
}

impl HcaDecoder<File> {
    /// Create a new HCA decoder from a file path
    pub fn from_file(filename: &str) -> Result<Self, HcaDecoderError> {
        let file = File::open(filename)?;
        let mut decoder = HcaDecoder::from_reader(file)?;
        decoder.owns_file = true;
        Ok(decoder)
    }
}

impl<R: Read + Seek> HcaDecoder<R> {
    /// Create a new HCA decoder from a reader
    pub fn from_reader(mut reader: R) -> Result<Self, HcaDecoderError> {
        // Test header
        let mut header_buf = [0u8; 8];
        reader.read_exact(&mut header_buf)?;

        let header_size = ClHca::is_hca_file(&header_buf).ok_or(HcaDecoderError::InvalidHeader)?;

        if header_size > 0x1000 {
            return Err(HcaDecoderError::InvalidHeader);
        }

        // Read full header
        let mut full_header = vec![0u8; header_size];
        reader.seek(SeekFrom::Start(0))?;
        reader.read_exact(&mut full_header)?;

        // Initialize decoder
        let mut handle = Box::new(ClHca::new());
        handle.decode_header(&full_header)?;

        let info = handle.get_info()?;

        // Allocate buffers
        let buf = vec![0u8; info.block_size as usize];
        let fbuf = vec![0.0f32; info.channel_count as usize * info.samples_per_block];

        let current_delay = info.encoder_delay as i32;

        Ok(Self {
            reader,
            info,
            handle,
            buf,
            chunk: Vec::new(),
            chunk_first: u32::MAX,
            chunk_blocks: 0,
            fbuf,
            current_delay,
            current_block: 0,
            reader_offset: Some(header_size as u64),
            owns_file: false,
            samples_written: 0,
        })
    }

    /// Reset the decoder to the beginning
    pub fn reset(&mut self) {
        self.handle.decode_reset();
        self.current_block = 0;
        self.current_delay = self.info.encoder_delay as i32;
        self.reader_offset = None;
        self.samples_written = 0;
    }

    /// Get the HCA file information
    pub fn info(&self) -> &HcaInfo {
        &self.info
    }

    /// CRI's canonical valid sample-frame count:
    /// block_count*samples_per_block - encoder_delay - encoder_padding (hca.h:56).
    fn total_valid_samples(&self) -> u64 {
        (self.info.block_count as u64 * self.info.samples_per_block as u64)
            .saturating_sub(self.info.encoder_delay as u64)
            .saturating_sub(self.info.encoder_padding as u64)
    }

    /// Set the decryption key
    pub fn set_encryption_key(&mut self, keycode: u64, subkey: u64) {
        let key = if subkey != 0 {
            keycode.wrapping_mul((subkey << 16) | (!subkey as u16 as u64).wrapping_add(2))
        } else {
            keycode
        };
        self.handle.set_key(key);
    }

    /// Read a single HCA frame/block, served from the read-ahead chunk buffer.
    fn read_packet(&mut self) -> Result<(), HcaDecoderError> {
        if self.current_block >= self.info.block_count {
            return Err(HcaDecoderError::Eof);
        }

        let block_size = self.info.block_size as usize;
        let in_chunk = self.current_block >= self.chunk_first
            && self.current_block - self.chunk_first < self.chunk_blocks;
        if !in_chunk {
            // Refill: read up to ~1 MiB of consecutive blocks in one syscall.
            let max_blocks = ((1 << 20) / block_size).clamp(1, 4096) as u32;
            let want = max_blocks.min(self.info.block_count - self.current_block);
            let offset = self.info.header_size as u64
                + self.current_block as u64 * self.info.block_size as u64;
            if self.reader_offset != Some(offset) {
                self.reader.seek(SeekFrom::Start(offset))?;
            }
            self.chunk.resize(want as usize * block_size, 0);
            self.chunk_first = u32::MAX;
            self.chunk_blocks = 0;
            self.reader_offset = None;
            self.reader.read_exact(&mut self.chunk)?;
            self.reader_offset = Some(offset + self.chunk.len() as u64);
            self.chunk_first = self.current_block;
            self.chunk_blocks = want;
        }

        let idx = (self.current_block - self.chunk_first) as usize * block_size;
        self.buf.copy_from_slice(&self.chunk[idx..idx + block_size]);
        self.current_block += 1;
        Ok(())
    }

    /// Decode a single frame and return the samples
    /// Returns (samples slice, num samples) or error
    pub fn decode_frame(&mut self) -> Result<(&[f32], usize), HcaDecoderError> {
        // Read packet
        self.read_packet()?;

        // Decode frame
        self.handle.decode_block(&mut self.buf)?;

        // Read samples
        self.handle.read_samples(&mut self.fbuf);

        let samples = self.info.samples_per_block as i32;
        let mut discard = 0;

        // Handle encoder delay
        if self.current_delay > 0 {
            if self.current_delay >= samples {
                self.current_delay -= samples;
                return Ok((&[], 0));
            }
            discard = self.current_delay;
            self.current_delay = 0;
        }

        let start_idx = discard as usize * self.info.channel_count as usize;
        let mut num_samples = (samples - discard) as usize;
        // Drop trailing encoder_padding: cap cumulative output at the valid count.
        let remaining = self
            .total_valid_samples()
            .saturating_sub(self.samples_written);
        if num_samples as u64 > remaining {
            num_samples = remaining as usize;
        }
        self.samples_written += num_samples as u64;
        Ok((&self.fbuf[start_idx..], num_samples))
    }

    /// Decode a single frame into interleaved 16-bit PCM samples.
    ///
    /// Returns the number of sample frames written, not the number of i16
    /// values. The caller-provided buffer must fit one full decoded HCA frame.
    pub fn decode_frame_i16(&mut self, pcm: &mut [i16]) -> Result<usize, HcaDecoderError> {
        let frame_len = self.info.samples_per_block * self.info.channel_count as usize;
        if pcm.len() < frame_len {
            return Err(HcaDecoderError::InvalidSampleRange);
        }

        self.read_packet()?;
        self.handle.decode_block(&mut self.buf)?;
        self.handle.read_samples_16(&mut pcm[..frame_len]);

        let samples = self.info.samples_per_block as i32;
        let mut discard = 0;

        if self.current_delay > 0 {
            if self.current_delay >= samples {
                self.current_delay -= samples;
                return Ok(0);
            }
            discard = self.current_delay;
            self.current_delay = 0;
        }

        let start = discard as usize * self.info.channel_count as usize;
        if start > 0 {
            pcm.copy_within(start..frame_len, 0);
        }

        let mut num_samples = (samples - discard) as usize;
        // Drop trailing encoder_padding: cap cumulative output at the valid count.
        let remaining = self
            .total_valid_samples()
            .saturating_sub(self.samples_written);
        if num_samples as u64 > remaining {
            num_samples = remaining as usize;
        }
        self.samples_written += num_samples as u64;
        Ok(num_samples)
    }

    /// Decode the entire HCA file and return all samples
    pub fn decode_all(&mut self) -> Result<Vec<f32>, HcaDecoderError> {
        self.reset();

        let channel_count = self.info.channel_count as usize;
        let total_samples = self.info.block_count as usize * self.info.samples_per_block;
        let mut all_samples = Vec::with_capacity(total_samples * channel_count);

        loop {
            match self.decode_frame() {
                Ok((samples, num_samples)) => {
                    let samples_to_add = num_samples * channel_count;
                    all_samples.extend_from_slice(&samples[..samples_to_add]);
                }
                Err(HcaDecoderError::Eof) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(all_samples)
    }

    /// Decode the entire HCA file as interleaved 16-bit PCM chunks.
    ///
    /// The callback receives only valid samples after encoder-delay trimming.
    /// This is useful when piping HCA directly into an audio encoder without
    /// materializing an intermediate WAV file.
    pub fn decode_to_pcm16_chunks<F>(&mut self, mut on_chunk: F) -> Result<(), HcaDecoderError>
    where
        F: FnMut(&[i16]) -> Result<(), HcaDecoderError>,
    {
        self.reset();

        let channels = self.info.channel_count as usize;
        let mut pcm_buf = vec![0i16; self.info.samples_per_block * channels];

        loop {
            match self.decode_frame_i16(&mut pcm_buf) {
                Ok(0) => {}
                Ok(sample_frames) => {
                    let sample_count = sample_frames * channels;
                    on_chunk(&pcm_buf[..sample_count])?;
                }
                Err(HcaDecoderError::Eof) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Seek to a specific sample position
    pub fn seek(&mut self, sample_num: u32) {
        let target_sample = sample_num + self.info.encoder_delay;
        let loop_start_block = target_sample / self.info.samples_per_block as u32;
        let loop_start_delay =
            target_sample - (loop_start_block * self.info.samples_per_block as u32);

        self.current_block = loop_start_block;
        self.current_delay = loop_start_delay as i32;
        self.reader_offset = None;
    }

    /// Test if a key correctly decrypts the HCA file
    pub fn test_key(&mut self, kt: &mut KeyTest) {
        let score = self.test_hca_score(kt);

        // Wrong key
        if score < 0 {
            return;
        }

        // Update if something better is found
        if kt.best_score <= 0 || (score < kt.best_score && score > 0) {
            kt.best_score = score;
            kt.best_key = kt.key;
        }
    }

    /// Test a number of frames to see if key decrypts correctly
    fn test_hca_score(&mut self, kt: &mut KeyTest) -> i32 {
        let mut test_frames = 0;
        let mut current_frame = 0u32;
        let mut blank_frames = 0;
        let mut total_score = 0;

        let mut offset = kt.start_offset;
        if offset == 0 {
            offset = self.info.header_size;
        }

        self.set_encryption_key(kt.key, kt.subkey);

        while test_frames < HCA_KEY_MAX_TEST_FRAMES && current_frame < self.info.block_count {
            let (score, should_break, new_offset) =
                self.test_single_frame(kt, offset, blank_frames);
            offset = new_offset;

            if should_break {
                total_score = -1;
                break;
            }

            if score < 0 {
                break;
            }

            current_frame += 1;

            if score == 0 && blank_frames < HCA_KEY_MAX_SKIP_BLANKS {
                blank_frames += 1;
                continue;
            }

            test_frames += 1;
            total_score += scale_frame_score(score);

            if total_score > HCA_KEY_MAX_TOTAL_SCORE {
                break;
            }
        }

        self.handle.decode_reset();
        finalize_score(total_score, test_frames)
    }

    fn test_single_frame(
        &mut self,
        kt: &mut KeyTest,
        offset: u32,
        _blank_frames: i32,
    ) -> (i32, bool, u32) {
        if self.reader.seek(SeekFrom::Start(offset as u64)).is_err() {
            return (-1, false, offset);
        }
        self.reader_offset = Some(offset as u64);

        if self.reader.read_exact(&mut self.buf).is_err() {
            return (-1, false, offset);
        }

        let score = self.handle.test_block(&mut self.buf);

        // Get first non-blank frame
        if kt.start_offset == 0 && score != 0 {
            kt.start_offset = offset;
        }

        let new_offset = offset + self.info.block_size;
        self.reader_offset = Some(new_offset as u64);

        if !(0..=HCA_KEY_MAX_FRAME_SCORE).contains(&score) {
            return (0, true, new_offset);
        }

        (score, false, new_offset)
    }

    /// Build the 44-byte RIFF/WAVE header for this file's PCM output.
    fn wav_header(&self, total_pcm_bytes: usize, smpl_len: usize) -> [u8; 44] {
        let mut header = [0u8; 44];
        header[0..4].copy_from_slice(b"RIFF");
        header[4..8].copy_from_slice(&((36 + total_pcm_bytes + smpl_len) as u32).to_le_bytes());
        header[8..12].copy_from_slice(b"WAVE");
        header[12..16].copy_from_slice(b"fmt ");
        header[16..20].copy_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        header[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM format
        header[22..24].copy_from_slice(&(self.info.channel_count as u16).to_le_bytes());
        header[24..28].copy_from_slice(&self.info.sampling_rate.to_le_bytes());
        let byte_rate = self.info.sampling_rate * self.info.channel_count * 2;
        header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
        let block_align = (self.info.channel_count * 2) as u16;
        header[32..34].copy_from_slice(&block_align.to_le_bytes());
        header[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits per sample
        header[36..40].copy_from_slice(b"data");
        header[40..44].copy_from_slice(&(total_pcm_bytes as u32).to_le_bytes());
        header
    }

    /// Decode the entire file to 16-bit WAV stream
    pub fn decode_to_wav<W: Write>(&mut self, w: &mut W) -> Result<(), HcaDecoderError> {
        self.reset();

        let total_samples = self.total_valid_samples() as usize;
        let total_pcm_bytes = total_samples * self.info.channel_count as usize * 2;

        // Optional WAV sampler (smpl) chunk carrying the HCA loop region.
        let smpl_chunk = self.loop_smpl_chunk();

        w.write_all(&self.wav_header(total_pcm_bytes, smpl_chunk.len()))?;

        let frame_len = self.info.samples_per_block * self.info.channel_count as usize;
        let mut data_buf = vec![0u8; frame_len * 2];

        self.decode_to_pcm16_chunks(|pcm| {
            if pcm.len() > frame_len {
                return Err(HcaDecoderError::InvalidSampleRange);
            }
            write_pcm_i16_le(w, pcm, &mut data_buf)?;
            Ok(())
        })?;

        if !smpl_chunk.is_empty() {
            w.write_all(&smpl_chunk)?;
        }
        Ok(())
    }

    /// Decode the entire file to 16-bit WAV using multiple threads.
    ///
    /// HCA blocks carry no cross-block state apart from the IMDCT overlap
    /// window (and, only for HCA v3 noise reconstruction, an RNG sequence), so
    /// the expensive part of each block — checksum, decrypt, bitstream
    /// unpack, dequantize, DCT stages — runs on `threads` worker threads while
    /// this thread performs the cheap sequential overlap-add, interleaving and
    /// writing, in block order. Output is byte-identical to `decode_to_wav`.
    ///
    /// Falls back to the serial `decode_to_wav` when `threads <= 1` or when
    /// the file is not block-parallelizable (HCA v3 with noise reconstruction).
    pub fn decode_to_wav_parallel<W: Write>(
        &mut self,
        w: &mut W,
        threads: usize,
    ) -> Result<(), HcaDecoderError> {
        use std::collections::BTreeMap;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc::sync_channel;

        if threads <= 1 || !self.handle.is_block_parallelizable() {
            return self.decode_to_wav(w);
        }

        self.reset();

        let channels = self.info.channel_count as usize;
        let samples_per_block = self.info.samples_per_block;
        let block_size = self.info.block_size as usize;
        let block_count = self.info.block_count as usize;
        let frame_f32 = channels * samples_per_block;

        let total_valid = self.total_valid_samples();
        let total_pcm_bytes = total_valid as usize * channels * 2;
        let smpl_chunk = self.loop_smpl_chunk();
        w.write_all(&self.wav_header(total_pcm_bytes, smpl_chunk.len()))?;

        // Load the whole audio region; workers slice blocks out of it.
        let mut data = vec![0u8; block_count * block_size];
        self.reader
            .seek(SeekFrom::Start(self.info.header_size as u64))?;
        self.reader_offset = None;
        self.reader.read_exact(&mut data)?;

        /// Blocks per work unit: big enough to amortize scheduling and the
        /// one duplicated seed-block decode per chunk, small enough for good
        /// load balancing and bounded in-flight memory.
        const CHUNK_BLOCKS: usize = 32;
        let num_chunks = block_count.div_ceil(CHUNK_BLOCKS);
        // No point spawning more workers than work units (each carries a
        // ClHca clone); also guards against absurd `threads` values.
        let threads = threads.min(num_chunks);
        let next_chunk = AtomicUsize::new(0);
        // Bounded so workers can't run unboundedly ahead of the writer.
        let (tx, rx) = sync_channel::<(usize, Result<Vec<u8>, HcaError>)>(threads * 2);

        let template = self.handle.as_ref().clone();
        let delay = self.info.encoder_delay as u64;

        std::thread::scope(|scope| -> Result<(), HcaDecoderError> {
            for _ in 0..threads {
                let tx = tx.clone();
                let next_chunk = &next_chunk;
                let data = &data[..];
                let mut hca = template.clone();
                // Explicit stack size: ClHca is ~170 KiB and debug-build
                // frames are large; the 2 MiB spawn default (absent
                // RUST_MIN_STACK) overflows on some targets.
                std::thread::Builder::new()
                    .name("hca-decode".into())
                    .stack_size(8 << 20)
                    .spawn_scoped(scope, move || {
                        let mut block = vec![0u8; block_size];
                        let mut dct = vec![0f32; frame_f32];
                        let mut prev = vec![[0f32; 128]; channels];
                        let mut wave = [0f32; 128];
                        let mut pcm = vec![0i16; frame_f32];
                        let mut scratch = vec![0u8; frame_f32 * 2];
                        loop {
                            let c = next_chunk.fetch_add(1, Ordering::Relaxed);
                            if c >= num_chunks {
                                break;
                            }
                            let lo = c * CHUNK_BLOCKS;
                            let hi = (lo + CHUNK_BLOCKS).min(block_count);
                            let payload = decode_chunk_pcm(DecodeChunkArgs {
                                hca: &mut hca,
                                data,
                                lo,
                                hi,
                                block_size,
                                channels,
                                samples_per_block,
                                delay,
                                total_valid,
                                block: &mut block,
                                dct: &mut dct,
                                prev: &mut prev,
                                wave: &mut wave,
                                pcm: &mut pcm,
                                scratch: &mut scratch,
                            });
                            if tx.send((c, payload)).is_err() {
                                break; // writer bailed out
                            }
                        }
                    })
                    .expect("spawn HCA decode worker");
            }
            drop(tx);

            // Writer: emit chunk byte payloads strictly in order.
            let mut pending: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
            for expect in 0..num_chunks {
                let bytes = loop {
                    if let Some(bytes) = pending.remove(&expect) {
                        break bytes;
                    }
                    // Channel can only disconnect early if a worker panicked.
                    let (c, payload) = rx
                        .recv()
                        .map_err(|_| HcaDecoderError::Hca(HcaError::InvalidParams))?;
                    let bytes = payload?;
                    if c == expect {
                        break bytes;
                    }
                    pending.insert(c, bytes);
                };
                w.write_all(&bytes)?;
            }
            Ok(())
        })?;

        self.current_block = self.info.block_count;
        self.samples_written = total_valid;
        let total_frames = self.info.block_count as u64 * samples_per_block as u64;
        self.current_delay = delay.saturating_sub(total_frames) as i32;

        if !smpl_chunk.is_empty() {
            w.write_all(&smpl_chunk)?;
        }
        Ok(())
    }

    /// Build a WAV `smpl` chunk for the HCA loop region, or empty if not looping.
    /// Loop samples per clhca.c: start = loop_start_block*spb + loop_start_delay
    /// - encoder_delay; end = loop_end_block*spb + (spb - loop_end_padding) - delay.
    fn loop_smpl_chunk(&self) -> Vec<u8> {
        if !self.info.loop_enabled {
            return Vec::new();
        }
        let spb = self.info.samples_per_block as u32;
        let delay = self.info.encoder_delay;
        let start = (self.info.loop_start_block * spb)
            .saturating_add(self.info.loop_start_delay)
            .saturating_sub(delay);
        let end = (self.info.loop_end_block * spb)
            .saturating_add(spb.saturating_sub(self.info.loop_end_padding))
            .saturating_sub(delay);
        let sample_period = 1_000_000_000u32
            .checked_div(self.info.sampling_rate)
            .unwrap_or(0);
        let mut c = Vec::with_capacity(68);
        c.extend_from_slice(b"smpl");
        c.extend_from_slice(&60u32.to_le_bytes()); // chunk body size
                                                   // manufacturer, product, sample_period, midi_unity_note, midi_pitch_fraction,
                                                   // smpte_format, smpte_offset, num_sample_loops, sampler_data
        for v in [0u32, 0, sample_period, 60, 0, 0, 0, 1, 0] {
            c.extend_from_slice(&v.to_le_bytes());
        }
        // one forward loop: identifier, type, start, end, fraction, play_count
        for v in [0u32, 0, start, end, 0, 0] {
            c.extend_from_slice(&v.to_le_bytes());
        }
        c
    }
}

/// Scratch and parameters for `decode_chunk_pcm`, grouped so the worker loop
/// reuses all buffers across chunks.
struct DecodeChunkArgs<'a> {
    hca: &'a mut ClHca,
    data: &'a [u8],
    lo: usize,
    hi: usize,
    block_size: usize,
    channels: usize,
    samples_per_block: usize,
    /// encoder_delay in sample frames.
    delay: u64,
    /// Total valid sample frames in the whole file.
    total_valid: u64,
    block: &'a mut [u8],
    dct: &'a mut [f32],
    prev: &'a mut [[f32; 128]],
    wave: &'a mut [f32; 128],
    pcm: &'a mut [i16],
    scratch: &'a mut [u8],
}

/// Decode blocks `lo..hi` to final interleaved 16-bit PCM (LE bytes), with
/// encoder delay/padding trimmed. Fully independent of other chunks: the IMDCT
/// overlap state is a pure function of the preceding subframe's DCT output, so
/// it is seeded by decoding block `lo - 1` (zeros at file start).
fn decode_chunk_pcm(a: DecodeChunkArgs<'_>) -> Result<Vec<u8>, HcaError> {
    let spb = a.samples_per_block;
    let frame_f32 = a.channels * spb;
    let mut bytes = Vec::with_capacity((a.hi - a.lo) * frame_f32 * 2);

    // Seed the overlap state.
    if a.lo == 0 {
        for p in a.prev.iter_mut() {
            p.fill(0.0);
        }
    } else {
        let b = a.lo - 1;
        a.block
            .copy_from_slice(&a.data[b * a.block_size..(b + 1) * a.block_size]);
        a.hca.decode_block_dct(a.block, a.dct)?;
        for (ch, prev_ch) in a.prev.iter_mut().enumerate() {
            let last = &a.dct[ch * spb + (HCA_SUBFRAMES - 1) * 128..][..128];
            // imdct_overlap's `previous` output depends only on `dct`.
            imdct_overlap(last.try_into().unwrap(), prev_ch, a.wave);
        }
    }

    let mut wave2 = [0f32; 128];
    for b in a.lo..a.hi {
        a.block
            .copy_from_slice(&a.data[b * a.block_size..(b + 1) * a.block_size]);
        a.hca.decode_block_dct(a.block, a.dct)?;
        for sf in 0..HCA_SUBFRAMES {
            if a.channels == 2 {
                // Stereo: overlap both channels, then a pairwise interleave
                // that the compiler can vectorize (no strided stores).
                let dct0: &[f32; 128] = a.dct[sf * 128..][..128].try_into().unwrap();
                let dct1: &[f32; 128] = a.dct[spb + sf * 128..][..128].try_into().unwrap();
                imdct_overlap(dct0, &mut a.prev[0], a.wave);
                imdct_overlap(dct1, &mut a.prev[1], &mut wave2);
                let out = &mut a.pcm[sf * 256..(sf + 1) * 256];
                for (pair, (&l, &r)) in out
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(a.wave.iter().zip(wave2.iter()))
                {
                    pair[0] = pcm_f32_to_i16(l);
                    pair[1] = pcm_f32_to_i16(r);
                }
            } else {
                for (ch, prev_ch) in a.prev.iter_mut().enumerate() {
                    let dct: &[f32; 128] = a.dct[ch * spb + sf * 128..][..128].try_into().unwrap();
                    imdct_overlap(dct, prev_ch, a.wave);
                    let base = sf * 128 * a.channels + ch;
                    for (j, &v) in a.wave.iter().enumerate() {
                        a.pcm[base + j * a.channels] = pcm_f32_to_i16(v);
                    }
                }
            }
        }

        // Static delay/padding trim: this block covers pre-trim sample frames
        // [b*spb, (b+1)*spb); the valid output range is [delay, delay+total_valid).
        let block_start = b as u64 * spb as u64;
        let keep_lo = a.delay.saturating_sub(block_start).min(spb as u64) as usize;
        let keep_hi = (a.delay + a.total_valid)
            .saturating_sub(block_start)
            .min(spb as u64) as usize;
        if keep_hi > keep_lo {
            // Writing into a Vec<u8> cannot fail.
            write_pcm_i16_le(
                &mut bytes,
                &a.pcm[keep_lo * a.channels..keep_hi * a.channels],
                a.scratch,
            )
            .expect("Vec write");
        }
    }
    Ok(bytes)
}

#[cfg(target_endian = "little")]
fn write_pcm_i16_le<W: Write>(
    writer: &mut W,
    samples: &[i16],
    _scratch: &mut [u8],
) -> io::Result<()> {
    let bytes = unsafe {
        // SAFETY: i16 is a plain integer type, and on little-endian targets its
        // in-memory representation is exactly the little-endian PCM byte order.
        std::slice::from_raw_parts(
            samples.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    };
    writer.write_all(bytes)
}

#[cfg(not(target_endian = "little"))]
fn write_pcm_i16_le<W: Write>(
    writer: &mut W,
    samples: &[i16],
    scratch: &mut [u8],
) -> io::Result<()> {
    let byte_len = samples.len() * 2;
    for (chunk, &sample) in scratch[..byte_len].chunks_exact_mut(2).zip(samples) {
        chunk.copy_from_slice(&sample.to_le_bytes());
    }
    writer.write_all(&scratch[..byte_len])
}

fn scale_frame_score(score: i32) -> i32 {
    match score {
        1 => 1,
        0 => 3 * HCA_KEY_SCORE_SCALE,
        _ => score * HCA_KEY_SCORE_SCALE,
    }
}

fn finalize_score(total_score: i32, test_frames: i32) -> i32 {
    // Signal best possible score
    if test_frames > HCA_KEY_MIN_TEST_FRAMES && total_score > 0 && total_score <= test_frames {
        return 1;
    }
    total_score
}

/// Errors that can occur during HCA decoding
#[derive(Debug)]
pub enum HcaDecoderError {
    Io(io::Error),
    Hca(HcaError),
    InvalidHeader,
    InvalidSampleRange,
    Eof,
}

impl std::fmt::Display for HcaDecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Hca(e) => write!(f, "HCA error: {}", e),
            Self::InvalidHeader => write!(f, "Invalid HCA header"),
            Self::InvalidSampleRange => write!(f, "Invalid sample range"),
            Self::Eof => write!(f, "End of file"),
        }
    }
}

impl std::error::Error for HcaDecoderError {}

impl From<io::Error> for HcaDecoderError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<HcaError> for HcaDecoderError {
    fn from(e: HcaError) -> Self {
        Self::Hca(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_frame_score() {
        assert_eq!(scale_frame_score(1), 1);
        assert_eq!(scale_frame_score(0), 3 * HCA_KEY_SCORE_SCALE); // 30
        assert_eq!(scale_frame_score(5), 5 * HCA_KEY_SCORE_SCALE); // 50
        assert_eq!(scale_frame_score(-1), -HCA_KEY_SCORE_SCALE); // -10
    }

    #[test]
    fn test_finalize_score() {
        // Best possible: enough frames, small positive score
        assert_eq!(finalize_score(4, 5), 1); // total_score(4) <= test_frames(5), frames > 3
                                             // Not enough frames
        assert_eq!(finalize_score(2, 2), 2); // test_frames(2) <= MIN_TEST_FRAMES(3)
                                             // Score too high
        assert_eq!(finalize_score(100, 5), 100);
        // Negative
        assert_eq!(finalize_score(-1, 5), -1);
    }

    #[test]
    fn test_key_test_default() {
        let kt = KeyTest::default();
        assert_eq!(kt.key, 0);
        assert_eq!(kt.subkey, 0);
        assert_eq!(kt.start_offset, 0);
        assert_eq!(kt.best_score, 0);
        assert_eq!(kt.best_key, 0);
    }

    #[test]
    fn test_hca_decoder_error_display() {
        let err = HcaDecoderError::InvalidHeader;
        assert_eq!(format!("{}", err), "Invalid HCA header");

        let err = HcaDecoderError::Eof;
        assert_eq!(format!("{}", err), "End of file");

        let err = HcaDecoderError::InvalidSampleRange;
        assert_eq!(format!("{}", err), "Invalid sample range");
    }
}
