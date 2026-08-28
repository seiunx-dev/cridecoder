//! HCA decoder core - header parsing and block decoding

use crate::hca::ath::ath_init;
use crate::hca::bitreader::BitReader;
use crate::hca::cipher::{cipher_decrypt, cipher_init};
use crate::hca::imdct::imdct_transform;
use crate::hca::tables::{
    DEQUANTIZER_RANGE_TABLE, DEQUANTIZER_SCALING_TABLE, INTENSITY_RATIO_TABLE, INVERT_TABLE,
    MAX_BIT_TABLE, READ_VAL_TABLE, SCALE_CONVERSION_TABLE,
};
use thiserror::Error;

// HCA version constants
pub const HCA_VERSION_101: u32 = 0x0101;
pub const HCA_VERSION_102: u32 = 0x0102;
pub const HCA_VERSION_103: u32 = 0x0103;
pub const HCA_VERSION_200: u32 = 0x0200;
pub const HCA_VERSION_300: u32 = 0x0300;

// HCA format constants
pub const HCA_MIN_FRAME_SIZE: u32 = 0x8;
pub const HCA_MAX_FRAME_SIZE: u32 = 0xFFFF;
pub const HCA_MASK: u32 = 0x7F7F7F7F;
pub const HCA_SUBFRAMES: usize = 8;
pub const HCA_SAMPLES_PER_SUBFRAME: usize = 128;
pub const HCA_SAMPLES_PER_FRAME: usize = HCA_SUBFRAMES * HCA_SAMPLES_PER_SUBFRAME;
pub const HCA_MIN_CHANNELS: u32 = 1;
pub const HCA_MAX_CHANNELS: usize = 16;
pub const HCA_MIN_SAMPLE_RATE: u32 = 1;
pub const HCA_MAX_SAMPLE_RATE: u32 = 0x7FFFFF;
pub const HCA_DEFAULT_RANDOM: u32 = 1;

/// HCA decoder errors
#[derive(Debug, Error)]
pub enum HcaError {
    #[error("invalid parameters")]
    InvalidParams,
    #[error("invalid HCA header")]
    InvalidHeader,
    #[error("checksum failed")]
    ChecksumFailed,
    #[error("sync word not found")]
    SyncError,
    #[error("unpack error: {0}")]
    UnpackError(String),
    #[error("bitreader error")]
    BitreaderError,
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("decoder not initialized")]
    NotInitialized,
}

/// Channel types for stereo processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelType {
    #[default]
    Discrete,
    StereoPrimary,
    StereoSecondary,
}

fn set_stereo_pair(channel_types: &mut [ChannelType], primary: usize) {
    channel_types[primary] = ChannelType::StereoPrimary;
    channel_types[primary + 1] = ChannelType::StereoSecondary;
}

pub(crate) fn assign_stereo_channel_types(
    channel_types: &mut [ChannelType],
    channels_per_track: usize,
    channel_config: u32,
) {
    set_stereo_pair(channel_types, 0);
    match channels_per_track {
        4 if channel_config == 0 => set_stereo_pair(channel_types, 2),
        5 if channel_config <= 2 => set_stereo_pair(channel_types, 3),
        6 | 7 => set_stereo_pair(channel_types, 4),
        8 => {
            set_stereo_pair(channel_types, 4);
            set_stereo_pair(channel_types, 6);
        }
        _ => {}
    }
}

/// Channel state for decoding
#[derive(Clone)]
pub struct StChannel {
    pub channel_type: ChannelType,
    pub coded_count: usize,

    pub intensity: [u8; HCA_SUBFRAMES],
    pub scale_factors: [u8; HCA_SAMPLES_PER_SUBFRAME],
    pub resolution: [u8; HCA_SAMPLES_PER_SUBFRAME],
    pub noises: [u8; HCA_SAMPLES_PER_SUBFRAME],
    pub noise_count: usize,
    pub valid_count: usize,

    pub gain: [f32; HCA_SAMPLES_PER_SUBFRAME],
    pub spectra: [[f32; HCA_SAMPLES_PER_SUBFRAME]; HCA_SUBFRAMES],
    pub temp: [f32; HCA_SAMPLES_PER_SUBFRAME],
    pub dct: [f32; HCA_SAMPLES_PER_SUBFRAME],
    pub imdct_previous: [f32; HCA_SAMPLES_PER_SUBFRAME],
    pub wave: [[f32; HCA_SAMPLES_PER_SUBFRAME]; HCA_SUBFRAMES],
}

impl Default for StChannel {
    fn default() -> Self {
        Self {
            channel_type: ChannelType::Discrete,
            coded_count: 0,
            intensity: [0; HCA_SUBFRAMES],
            scale_factors: [0; HCA_SAMPLES_PER_SUBFRAME],
            resolution: [0; HCA_SAMPLES_PER_SUBFRAME],
            noises: [0; HCA_SAMPLES_PER_SUBFRAME],
            noise_count: 0,
            valid_count: 0,
            gain: [0.0; HCA_SAMPLES_PER_SUBFRAME],
            spectra: [[0.0; HCA_SAMPLES_PER_SUBFRAME]; HCA_SUBFRAMES],
            temp: [0.0; HCA_SAMPLES_PER_SUBFRAME],
            dct: [0.0; HCA_SAMPLES_PER_SUBFRAME],
            imdct_previous: [0.0; HCA_SAMPLES_PER_SUBFRAME],
            wave: [[0.0; HCA_SAMPLES_PER_SUBFRAME]; HCA_SUBFRAMES],
        }
    }
}

/// HCA decoder information
#[derive(Debug, Clone)]
pub struct HcaInfo {
    pub version: u32,
    pub header_size: u32,
    pub sampling_rate: u32,
    pub channel_count: u32,
    pub block_size: u32,
    pub block_count: u32,
    pub encoder_delay: u32,
    pub encoder_padding: u32,
    pub loop_enabled: bool,
    pub loop_start_block: u32,
    pub loop_end_block: u32,
    pub loop_start_delay: u32,
    pub loop_end_padding: u32,
    pub samples_per_block: usize,
    pub comment: String,
    pub encryption_enabled: bool,
}

/// CRC16 table for HCA
pub const CRC16_TABLE: [u16; 256] = [
    0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011, 0x8033, 0x0036, 0x003C, 0x8039,
    0x0028, 0x802D, 0x8027, 0x0022, 0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072,
    0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041, 0x80C3, 0x00C6, 0x00CC, 0x80C9,
    0x00D8, 0x80DD, 0x80D7, 0x00D2, 0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1,
    0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1, 0x8093, 0x0096, 0x009C, 0x8099,
    0x0088, 0x808D, 0x8087, 0x0082, 0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192,
    0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1, 0x01E0, 0x81E5, 0x81EF, 0x01EA,
    0x81FB, 0x01FE, 0x01F4, 0x81F1, 0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2,
    0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151, 0x8173, 0x0176, 0x017C, 0x8179,
    0x0168, 0x816D, 0x8167, 0x0162, 0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
    0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101, 0x8303, 0x0306, 0x030C, 0x8309,
    0x0318, 0x831D, 0x8317, 0x0312, 0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321,
    0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371, 0x8353, 0x0356, 0x035C, 0x8359,
    0x0348, 0x834D, 0x8347, 0x0342, 0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1,
    0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2, 0x83A3, 0x03A6, 0x03AC, 0x83A9,
    0x03B8, 0x83BD, 0x83B7, 0x03B2, 0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381,
    0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291, 0x82B3, 0x02B6, 0x02BC, 0x82B9,
    0x02A8, 0x82AD, 0x82A7, 0x02A2, 0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2,
    0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1, 0x8243, 0x0246, 0x024C, 0x8249,
    0x0258, 0x825D, 0x8257, 0x0252, 0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261,
    0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231, 0x8213, 0x0216, 0x021C, 0x8219,
    0x0208, 0x820D, 0x8207, 0x0202,
];

/// Calculate CRC16 checksum for HCA data
/// Slicing-by-8 tables derived from CRC16_TABLE. T[k][i] = CRC of byte `i`
/// followed by `k` zero bytes (T[0] == CRC16_TABLE), for the MSB-first CRC.
const CRC16_SLICE: [[u16; 256]; 8] = build_crc16_slice();

const fn build_crc16_slice() -> [[u16; 256]; 8] {
    let mut t = [[0u16; 256]; 8];
    let mut i = 0;
    while i < 256 {
        t[0][i] = CRC16_TABLE[i];
        i += 1;
    }
    let mut k = 1;
    while k < 8 {
        let mut j = 0;
        while j < 256 {
            let prev = t[k - 1][j];
            t[k][j] = (prev << 8) ^ CRC16_TABLE[(prev >> 8) as usize];
            j += 1;
        }
        k += 1;
    }
    t
}

/// MSB-first CRC16, slicing-by-8: consumes 8 bytes per step via parallel table
/// lookups, with a byte-at-a-time tail. Produces the identical checksum to the
/// naive loop (verified in tests) but ~5x faster over a full frame.
pub fn crc16_checksum(data: &[u8]) -> u16 {
    let mut sum: u16 = 0;
    let (chunks, remainder) = data.as_chunks::<8>();
    for c in chunks {
        let i0 = ((sum >> 8) as u8 ^ c[0]) as usize;
        let i1 = (sum as u8 ^ c[1]) as usize;
        sum = CRC16_SLICE[7][i0]
            ^ CRC16_SLICE[6][i1]
            ^ CRC16_SLICE[5][c[2] as usize]
            ^ CRC16_SLICE[4][c[3] as usize]
            ^ CRC16_SLICE[3][c[4] as usize]
            ^ CRC16_SLICE[2][c[5] as usize]
            ^ CRC16_SLICE[1][c[6] as usize]
            ^ CRC16_SLICE[0][c[7] as usize];
    }
    for &byte in remainder {
        sum = (sum << 8) ^ CRC16_TABLE[((sum >> 8) ^ byte as u16) as usize];
    }
    sum
}

#[cfg(test)]
mod crc_tests {
    use super::*;

    fn crc16_naive(data: &[u8]) -> u16 {
        let mut sum: u16 = 0;
        for &byte in data {
            sum = (sum << 8) ^ CRC16_TABLE[((sum >> 8) ^ byte as u16) as usize];
        }
        sum
    }

    #[test]
    fn test_crc16_slice8_matches_naive() {
        // Every length class mod 8, plus a full-frame-sized pseudo-random payload.
        for len in 0..40usize {
            let data: Vec<u8> = (0..len)
                .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
                .collect();
            assert_eq!(crc16_checksum(&data), crc16_naive(&data), "len {len}");
        }
        let big: Vec<u8> = (0..682u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        assert_eq!(crc16_checksum(&big), crc16_naive(&big));
    }
}

fn header_ceil2(a: u32, b: u32) -> u32 {
    if b < 1 {
        return 0;
    }
    let mut result = a / b;
    if !a.is_multiple_of(b) {
        result += 1;
    }
    result
}

fn sample_quality(sample: f32, scale: f32) -> (i32, i32) {
    if !(-1.0..=1.0).contains(&sample) {
        return (1, 0);
    }
    let pcm = (sample * scale) as i32;
    (0, i32::from(pcm == 0 || pcm == -1))
}

fn decode_intensity_delta(
    br: &mut BitReader,
    value: u8,
    delta: u8,
    maximum_delta: u8,
) -> Result<u8, HcaError> {
    if delta == maximum_delta {
        return Ok(br.read(4) as u8);
    }
    let value = value.wrapping_sub(maximum_delta >> 1).wrapping_add(delta);
    if value > 15 {
        return Err(HcaError::UnpackError("invalid intensity".into()));
    }
    Ok(value)
}

fn resolution_for_scale_factor(
    scale_factor: u8,
    ath: u8,
    packed_noise_level: u32,
    band: usize,
    minimum: u8,
    maximum: u8,
) -> u8 {
    if scale_factor == 0 {
        return 0;
    }

    // clHCA keeps packed_noise_level unsigned: the shift must remain logical.
    let noise_level = ath as i32 + ((packed_noise_level.wrapping_add(band as u32) >> 8) as i32);
    let curve_position = noise_level + 1 - ((5 * scale_factor as i32) >> 1);
    let resolution = match curve_position {
        ..0 => 15,
        0..=65 => INVERT_TABLE[curve_position as usize],
        _ => 0,
    };
    resolution.clamp(minimum, maximum)
}

/// Consumed-bit-count model for `dequantize_coefficients`: every
/// READ_BIT_TABLE row is "the first DEQ_USED_THRESH[res] codes take
/// DEQ_USED_BASE[res] bits, the rest take one more", and sign-magnitude
/// resolutions (8..=15) consume MAX_BIT_TABLE[res]-1 bits when the magnitude
/// is 0 (code < 2). Equivalence with READ_BIT_TABLE is asserted in tests.
const DEQ_USED_BASE: [u8; 16] = [0, 1, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const DEQ_USED_THRESH: [u8; 16] = [1, 2, 6, 2, 14, 10, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2];

/// Main HCA decoder structure
#[derive(Clone)]
pub struct ClHca {
    is_valid: bool,

    // Header config
    pub version: u32,
    pub header_size: u32,
    pub channels: u32,
    pub sample_rate: u32,
    pub frame_count: u32,
    pub encoder_delay: u32,
    pub encoder_padding: u32,
    pub frame_size: u32,
    pub min_resolution: u32,
    pub max_resolution: u32,
    pub track_count: u32,
    pub channel_config: u32,
    pub stereo_type: u32,
    pub total_band_count: u32,
    pub base_band_count: u32,
    pub stereo_band_count: u32,
    pub bands_per_hfr_group: u32,
    pub ms_stereo: u32,
    pub reserved: u32,

    pub vbr_max_frame_size: u32,
    pub vbr_noise_level: u32,

    pub ath_type: u32,

    pub loop_start_frame: u32,
    pub loop_end_frame: u32,
    pub loop_start_delay: u32,
    pub loop_end_padding: u32,
    pub loop_flag: bool,

    pub ciph_type: u32,
    pub keycode: u64,

    pub rva_volume: f32,

    pub comment_len: usize,
    pub comment: [u8; 256],

    // State
    pub hfr_group_count: u32,
    pub ath_curve: [u8; HCA_SAMPLES_PER_SUBFRAME],
    pub cipher_table: [u8; 256],
    pub random: u32,
    pub channel: [StChannel; HCA_MAX_CHANNELS],
}

impl Default for ClHca {
    fn default() -> Self {
        Self::new()
    }
}

impl ClHca {
    /// Create a new HCA decoder instance
    pub fn new() -> Self {
        Self {
            is_valid: false,
            version: 0,
            header_size: 0,
            channels: 0,
            sample_rate: 0,
            frame_count: 0,
            encoder_delay: 0,
            encoder_padding: 0,
            frame_size: 0,
            min_resolution: 0,
            max_resolution: 0,
            track_count: 0,
            channel_config: 0,
            stereo_type: 0,
            total_band_count: 0,
            base_band_count: 0,
            stereo_band_count: 0,
            bands_per_hfr_group: 0,
            ms_stereo: 0,
            reserved: 0,
            vbr_max_frame_size: 0,
            vbr_noise_level: 0,
            ath_type: 0,
            loop_start_frame: 0,
            loop_end_frame: 0,
            loop_start_delay: 0,
            loop_end_padding: 0,
            loop_flag: false,
            ciph_type: 0,
            keycode: 0,
            rva_volume: 1.0,
            comment_len: 0,
            comment: [0; 256],
            hfr_group_count: 0,
            ath_curve: [0; HCA_SAMPLES_PER_SUBFRAME],
            cipher_table: [0; 256],
            random: HCA_DEFAULT_RANDOM,
            channel: std::array::from_fn(|_| StChannel::default()),
        }
    }

    /// Clear and reset the decoder
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Set decryption key
    pub fn set_key(&mut self, keycode: u64) {
        self.keycode = keycode;
        if self.is_valid {
            cipher_init(&mut self.cipher_table, self.ciph_type, self.keycode);
        }
    }

    /// Check if data is a valid HCA file
    pub fn is_hca_file(data: &[u8]) -> Option<usize> {
        if data.len() < 8 {
            return None;
        }

        let mut br = BitReader::new(data);
        let sig = br.peek(32) & HCA_MASK;

        if sig == 0x48434100 {
            // 'HCA\0'
            br.skip(32 + 16);
            let header_size = br.read(16) as usize;
            if header_size == 0 {
                return None;
            }
            Some(header_size)
        } else {
            None
        }
    }

    /// Get decoder information
    pub fn get_info(&self) -> Result<HcaInfo, HcaError> {
        if !self.is_valid {
            return Err(HcaError::NotInitialized);
        }

        Ok(HcaInfo {
            version: self.version,
            header_size: self.header_size,
            sampling_rate: self.sample_rate,
            channel_count: self.channels,
            block_size: self.frame_size,
            block_count: self.frame_count,
            encoder_delay: self.encoder_delay,
            encoder_padding: self.encoder_padding,
            loop_enabled: self.loop_flag,
            loop_start_block: self.loop_start_frame,
            loop_end_block: self.loop_end_frame,
            loop_start_delay: self.loop_start_delay,
            loop_end_padding: self.loop_end_padding,
            samples_per_block: HCA_SAMPLES_PER_FRAME,
            comment: if self.comment_len > 0 {
                String::from_utf8_lossy(&self.comment[..self.comment_len]).to_string()
            } else {
                String::new()
            },
            encryption_enabled: self.ciph_type == 56,
        })
    }

    /// Decode HCA header
    pub fn decode_header(&mut self, data: &[u8]) -> Result<(), HcaError> {
        if data.len() < 8 {
            return Err(HcaError::InvalidParams);
        }

        self.is_valid = false;

        let mut br = BitReader::new(data);

        self.decode_base_header(&mut br, data)?;
        self.decode_chunks(&mut br)?;
        self.validate_and_initialize()?;

        self.is_valid = true;
        Ok(())
    }

    fn decode_base_header(&mut self, br: &mut BitReader, data: &[u8]) -> Result<(), HcaError> {
        if (br.peek(32) & HCA_MASK) != 0x48434100 {
            return Err(HcaError::InvalidHeader);
        }

        br.skip(32);
        self.version = br.read(16);
        self.header_size = br.read(16);

        if self.version != HCA_VERSION_101
            && self.version != HCA_VERSION_102
            && self.version != HCA_VERSION_103
            && self.version != HCA_VERSION_200
            && self.version != HCA_VERSION_300
        {
            return Err(HcaError::UnsupportedVersion(self.version));
        }

        if data.len() < self.header_size as usize {
            return Err(HcaError::InvalidHeader);
        }

        if crc16_checksum(&data[..self.header_size as usize]) != 0 {
            return Err(HcaError::ChecksumFailed);
        }

        Ok(())
    }

    fn decode_chunks(&mut self, br: &mut BitReader) -> Result<(), HcaError> {
        self.decode_fmt_chunk(br)?;
        self.decode_comp_dec_chunk(br)?;
        self.decode_vbr_chunk(br)?;
        self.decode_ath_chunk(br);
        self.decode_loop_chunk(br)?;
        self.decode_cipher_chunk(br)?;
        self.decode_rva_chunk(br);
        self.decode_comment_chunk(br);
        Ok(())
    }

    fn decode_fmt_chunk(&mut self, br: &mut BitReader) -> Result<(), HcaError> {
        if (br.peek(32) & HCA_MASK) != 0x666D7400 {
            // "fmt\0"
            return Err(HcaError::InvalidHeader);
        }

        br.skip(32);
        self.channels = br.read(8);
        self.sample_rate = br.read(24);
        self.frame_count = br.read(32);
        self.encoder_delay = br.read(16);
        self.encoder_padding = br.read(16);

        if self.channels < HCA_MIN_CHANNELS || self.channels > HCA_MAX_CHANNELS as u32 {
            return Err(HcaError::InvalidHeader);
        }
        if self.frame_count == 0 {
            return Err(HcaError::InvalidHeader);
        }
        if self.sample_rate < HCA_MIN_SAMPLE_RATE || self.sample_rate > HCA_MAX_SAMPLE_RATE {
            return Err(HcaError::InvalidHeader);
        }

        Ok(())
    }

    fn decode_comp_dec_chunk(&mut self, br: &mut BitReader) -> Result<(), HcaError> {
        let chunk_type = br.peek(32) & HCA_MASK;

        if chunk_type == 0x636F6D70 {
            // "comp"
            self.decode_comp_chunk(br)
        } else if chunk_type == 0x64656300 {
            // "dec\0"
            self.decode_dec_chunk(br)
        } else {
            Err(HcaError::InvalidHeader)
        }
    }

    fn decode_comp_chunk(&mut self, br: &mut BitReader) -> Result<(), HcaError> {
        br.skip(32);
        self.frame_size = br.read(16);
        self.min_resolution = br.read(8);
        self.max_resolution = br.read(8);
        self.track_count = br.read(8);
        self.channel_config = br.read(8);
        self.total_band_count = br.read(8);
        self.base_band_count = br.read(8);
        self.stereo_band_count = br.read(8);
        self.bands_per_hfr_group = br.read(8);
        self.ms_stereo = br.read(8);
        self.reserved = br.read(8);
        Ok(())
    }

    fn decode_dec_chunk(&mut self, br: &mut BitReader) -> Result<(), HcaError> {
        br.skip(32);
        self.frame_size = br.read(16);
        self.min_resolution = br.read(8);
        self.max_resolution = br.read(8);
        self.total_band_count = br.read(8) + 1;
        self.base_band_count = br.read(8) + 1;
        self.track_count = br.read(4);
        self.channel_config = br.read(4);
        self.stereo_type = br.read(8);

        if self.stereo_type == 0 {
            self.base_band_count = self.total_band_count;
        }
        self.stereo_band_count = self.total_band_count - self.base_band_count;
        self.bands_per_hfr_group = 0;
        Ok(())
    }

    fn decode_vbr_chunk(&mut self, br: &mut BitReader) -> Result<(), HcaError> {
        if (br.peek(32) & HCA_MASK) == 0x76627200 {
            // "vbr\0"
            br.skip(32);
            self.vbr_max_frame_size = br.read(16);
            self.vbr_noise_level = br.read(16);
            // clhca.c: a vbr chunk requires frame_size==0 and 8 < max <= 0x1FF.
            if self.frame_size != 0
                || self.vbr_max_frame_size <= 8
                || self.vbr_max_frame_size > 0x1FF
            {
                return Err(HcaError::InvalidHeader);
            }
        } else {
            self.vbr_max_frame_size = 0;
            self.vbr_noise_level = 0;
        }
        Ok(())
    }

    fn decode_ath_chunk(&mut self, br: &mut BitReader) {
        if (br.peek(32) & HCA_MASK) == 0x61746800 {
            // "ath\0"
            br.skip(32);
            self.ath_type = br.read(16);
        } else {
            self.ath_type = if self.version < HCA_VERSION_200 { 1 } else { 0 };
        }
    }

    fn decode_loop_chunk(&mut self, br: &mut BitReader) -> Result<(), HcaError> {
        if (br.peek(32) & HCA_MASK) == 0x6C6F6F70 {
            // "loop"
            br.skip(32);
            self.loop_start_frame = br.read(32);
            self.loop_end_frame = br.read(32);
            self.loop_start_delay = br.read(16);
            self.loop_end_padding = br.read(16);
            self.loop_flag = true;

            if !(self.loop_start_frame <= self.loop_end_frame
                && self.loop_end_frame < self.frame_count)
            {
                return Err(HcaError::InvalidHeader);
            }
        } else {
            self.loop_flag = false;
        }
        Ok(())
    }

    fn decode_cipher_chunk(&mut self, br: &mut BitReader) -> Result<(), HcaError> {
        if (br.peek(32) & HCA_MASK) == 0x63697068 {
            // "ciph"
            br.skip(32);
            self.ciph_type = br.read(16);

            if !(self.ciph_type == 0 || self.ciph_type == 1 || self.ciph_type == 56) {
                return Err(HcaError::InvalidHeader);
            }
        } else {
            self.ciph_type = 0;
        }
        Ok(())
    }

    fn decode_rva_chunk(&mut self, br: &mut BitReader) {
        if (br.peek(32) & HCA_MASK) == 0x72766100 {
            // "rva\0"
            br.skip(32);
            let rva_int = br.read(32);
            self.rva_volume = f32::from_bits(rva_int);
        } else {
            self.rva_volume = 1.0;
        }
    }

    fn decode_comment_chunk(&mut self, br: &mut BitReader) {
        if (br.peek(32) & HCA_MASK) == 0x636F6D6D {
            // "comm"
            br.skip(32);
            self.comment_len = br.read(8) as usize;

            for i in 0..self.comment_len.min(255) {
                self.comment[i] = br.read(8) as u8;
            }
            if self.comment_len < 256 {
                self.comment[self.comment_len] = 0;
            }
        } else {
            self.comment_len = 0;
        }
    }

    fn validate_and_initialize(&mut self) -> Result<(), HcaError> {
        self.validate_frame_and_resolution()?;
        self.validate_tracks_and_bands()?;
        self.initialize_decoder_state()?;
        Ok(())
    }

    fn validate_frame_and_resolution(&self) -> Result<(), HcaError> {
        if self.frame_size < HCA_MIN_FRAME_SIZE || self.frame_size > HCA_MAX_FRAME_SIZE {
            return Err(HcaError::InvalidHeader);
        }

        if self.version <= HCA_VERSION_200 {
            if self.min_resolution != 1 || self.max_resolution != 15 {
                return Err(HcaError::InvalidHeader);
            }
        } else if self.min_resolution > self.max_resolution || self.max_resolution > 15 {
            return Err(HcaError::InvalidHeader);
        }

        Ok(())
    }

    fn validate_tracks_and_bands(&mut self) -> Result<(), HcaError> {
        if self.track_count == 0 {
            self.track_count = 1;
        }

        if self.track_count > self.channels {
            return Err(HcaError::InvalidHeader);
        }

        let max = HCA_SAMPLES_PER_SUBFRAME as u32;
        if self.total_band_count > max
            || self.base_band_count > max
            || self.stereo_band_count > max
            || self.base_band_count + self.stereo_band_count > max
            || self.bands_per_hfr_group > max
        {
            return Err(HcaError::InvalidHeader);
        }

        self.hfr_group_count = header_ceil2(
            self.total_band_count - self.base_band_count - self.stereo_band_count,
            self.bands_per_hfr_group,
        );

        Ok(())
    }

    fn initialize_decoder_state(&mut self) -> Result<(), HcaError> {
        if !ath_init(&mut self.ath_curve, self.ath_type, self.sample_rate) {
            return Err(HcaError::InvalidHeader);
        }

        cipher_init(&mut self.cipher_table, self.ciph_type, self.keycode);
        self.init_channels()?;
        self.random = HCA_DEFAULT_RANDOM;

        Ok(())
    }

    fn init_channels(&mut self) -> Result<(), HcaError> {
        let mut channel_types = [ChannelType::Discrete; HCA_MAX_CHANNELS];
        let channels_per_track = (self.channels / self.track_count) as usize;

        if self.stereo_band_count > 0 && channels_per_track > 1 {
            for i in 0..self.track_count as usize {
                let start = i * channels_per_track;
                let track_types = &mut channel_types[start..start + channels_per_track];
                assign_stereo_channel_types(track_types, channels_per_track, self.channel_config);
            }
        }

        for (i, &ct) in channel_types
            .iter()
            .enumerate()
            .take(self.channels as usize)
        {
            self.channel[i].channel_type = ct;

            if ct != ChannelType::StereoSecondary {
                self.channel[i].coded_count =
                    (self.base_band_count + self.stereo_band_count) as usize;
            } else {
                self.channel[i].coded_count = self.base_band_count as usize;
            }
        }

        Ok(())
    }

    /// Reset decoder state between files
    pub fn decode_reset(&mut self) {
        if !self.is_valid {
            return;
        }

        self.random = HCA_DEFAULT_RANDOM;

        for i in 0..self.channels as usize {
            self.channel[i].imdct_previous.fill(0.0);
        }
    }

    fn read_mono_samples_16(&self, samples: &mut [i16]) {
        for subframe in 0..HCA_SUBFRAMES {
            let base = subframe * HCA_SAMPLES_PER_SUBFRAME;
            for sample in 0..HCA_SAMPLES_PER_SUBFRAME {
                samples[base + sample] = pcm_f32_to_i16(self.channel[0].wave[subframe][sample]);
            }
        }
    }

    fn read_stereo_samples_16(&self, samples: &mut [i16]) {
        for subframe in 0..HCA_SUBFRAMES {
            let base = subframe * HCA_SAMPLES_PER_SUBFRAME * 2;
            for sample in 0..HCA_SAMPLES_PER_SUBFRAME {
                let output = base + sample * 2;
                samples[output] = pcm_f32_to_i16(self.channel[0].wave[subframe][sample]);
                samples[output + 1] = pcm_f32_to_i16(self.channel[1].wave[subframe][sample]);
            }
        }
    }

    fn read_interleaved_samples_16(&self, samples: &mut [i16], channels: usize) {
        let mut output = 0;
        for subframe in 0..HCA_SUBFRAMES {
            for sample in 0..HCA_SAMPLES_PER_SUBFRAME {
                for channel in self.channel.iter().take(channels) {
                    samples[output] = pcm_f32_to_i16(channel.wave[subframe][sample]);
                    output += 1;
                }
            }
        }
    }

    /// Read decoded samples as 16-bit PCM
    pub fn read_samples_16(&self, samples: &mut [i16]) {
        match self.channels as usize {
            1 => self.read_mono_samples_16(samples),
            2 => self.read_stereo_samples_16(samples),
            channels => self.read_interleaved_samples_16(samples, channels),
        }
    }

    /// Read decoded samples as f32
    pub fn read_samples(&self, samples: &mut [f32]) {
        let mut idx = 0;
        for i in 0..HCA_SUBFRAMES {
            for j in 0..HCA_SAMPLES_PER_SUBFRAME {
                for k in 0..self.channels as usize {
                    samples[idx] = self.channel[k].wave[i][j];
                    idx += 1;
                }
            }
        }
    }

    /// Test if a block decodes correctly (for key testing)
    /// Returns: <0 error/wrong, 0 unknown/silent, >0 good (closer to 1 is better)
    pub fn test_block(&mut self, data: &mut [u8]) -> i32 {
        // Check if block is empty
        if self.is_empty_block(data) {
            return 0;
        }

        // Try to unpack
        let bit_pos = match self.decode_block_unpack(data) {
            Ok(pos) => pos,
            Err(_) => return -1,
        };

        // Validate bitreader
        let err = self.validate_bitreader(data, bit_pos);
        if err != 0 {
            return err;
        }

        // Transform
        self.decode_block_transform();

        // Evaluate quality
        self.evaluate_decode_quality()
    }

    fn is_empty_block(&self, data: &[u8]) -> bool {
        for &byte in &data[0x02..(data.len().saturating_sub(0x02))] {
            if byte != 0 {
                return false;
            }
        }
        true
    }

    fn validate_bitreader(&self, data: &[u8], bit_pos: usize) -> i32 {
        let bits_max = self.frame_size as usize * 8;
        if bit_pos + 14 > bits_max {
            return -2; // bitreader error
        }

        let byte_start = if !bit_pos.is_multiple_of(8) {
            bit_pos / 8 + 1
        } else {
            bit_pos / 8
        };

        for &byte in &data[byte_start..(self.frame_size as usize).saturating_sub(0x02)] {
            if byte != 0 {
                return -1;
            }
        }

        0
    }

    fn evaluate_decode_quality(&self) -> i32 {
        const FRAME_SAMPLES: usize = HCA_SUBFRAMES * HCA_SAMPLES_PER_SUBFRAME;
        const SCALE: f32 = 32768.0;

        let mut clips = 0;
        let mut blanks = 0;
        let mut channel_blanks = [0i32; HCA_MAX_CHANNELS];

        for (ch, channel_blank) in channel_blanks
            .iter_mut()
            .enumerate()
            .take(self.channels as usize)
        {
            for sf in 0..HCA_SUBFRAMES {
                for s in 0..HCA_SAMPLES_PER_SUBFRAME {
                    let fsample = self.channel[ch].wave[sf][s];
                    let (sample_clips, sample_blanks) = sample_quality(fsample, SCALE);
                    clips += sample_clips;
                    blanks += sample_blanks;
                    *channel_blank += sample_blanks;
                }
            }
        }

        self.calculate_score(clips, blanks, &channel_blanks, FRAME_SAMPLES)
    }

    fn calculate_score(
        &self,
        mut clips: i32,
        blanks: i32,
        channel_blanks: &[i32],
        frame_samples: usize,
    ) -> i32 {
        if clips == 1 {
            clips += 1;
        }
        if clips > 1 {
            return clips;
        }

        if blanks == self.channels as i32 * frame_samples as i32 {
            return 0;
        }

        if self.channels >= 2
            && channel_blanks[0] == frame_samples as i32
            && channel_blanks[1] != frame_samples as i32
        {
            return 3;
        }

        1
    }

    /// Decode a block of HCA data
    pub fn decode_block(&mut self, data: &mut [u8]) -> Result<(), HcaError> {
        let _bit_pos = self.decode_block_unpack(data)?;
        self.decode_block_transform();
        Ok(())
    }

    /// True when blocks can be decoded independently of each other. Only
    /// files with `min_resolution == 0` (HCA v3 noise reconstruction) carry
    /// cross-block state, the sequential `random` LCG.
    pub fn is_block_parallelizable(&self) -> bool {
        self.is_valid && self.min_resolution > 0
    }

    /// Decode one block up to (and including) the DCT stages, without the
    /// sequential overlap-add, writing the per-channel per-subframe DCT-IV
    /// output into `out` (layout `[channel][subframe][sample]`, so
    /// `channels * 8 * 128` f32). Combined with `imdct_overlap` this splits a
    /// block decode into an independent parallel part and a cheap serial part.
    ///
    /// Errors on files where `is_block_parallelizable()` is false.
    pub fn decode_block_dct(&mut self, data: &mut [u8], out: &mut [f32]) -> Result<(), HcaError> {
        if !self.is_block_parallelizable() {
            return Err(HcaError::InvalidParams);
        }
        let channels = self.channels as usize;
        if out.len() < channels * HCA_SAMPLES_PER_FRAME {
            return Err(HcaError::InvalidParams);
        }

        self.decode_block_unpack(data)?;

        // Mirrors decode_block_transform, minus reconstruct_noise (a no-op
        // when min_resolution > 0) and minus the overlap-add.
        let full = self.bands_per_hfr_group != 0 || self.stereo_band_count != 0;
        for subframe in 0..HCA_SUBFRAMES {
            if full {
                self.restore_dct_bands(subframe, channels);
            }
            self.write_dct_subframe(subframe, channels, out);
        }
        Ok(())
    }

    fn restore_dct_bands(&mut self, subframe: usize, channels: usize) {
        for channel in 0..channels {
            self.reconstruct_high_frequency(channel, subframe);
        }
        if self.stereo_band_count == 0 {
            return;
        }
        for channel in 0..channels.saturating_sub(1) {
            self.apply_intensity_stereo(channel, subframe);
            self.apply_ms_stereo(channel, subframe);
        }
    }

    fn write_dct_subframe(&mut self, subframe: usize, channels: usize, out: &mut [f32]) {
        for (channel, chunk) in out
            .as_chunks_mut::<HCA_SAMPLES_PER_FRAME>()
            .0
            .iter_mut()
            .enumerate()
            .take(channels)
        {
            crate::hca::imdct::imdct_dct(&mut self.channel[channel], subframe);
            let start = subframe * HCA_SAMPLES_PER_SUBFRAME;
            chunk[start..start + HCA_SAMPLES_PER_SUBFRAME]
                .copy_from_slice(&self.channel[channel].spectra[subframe]);
        }
    }

    fn decode_block_unpack(&mut self, data: &mut [u8]) -> Result<usize, HcaError> {
        if !self.is_valid {
            return Err(HcaError::InvalidParams);
        }
        if data.len() < self.frame_size as usize {
            return Err(HcaError::InvalidParams);
        }

        let mut br = BitReader::new(data);

        // Test sync
        let sync = br.read(16);
        if sync != 0xFFFF {
            return Err(HcaError::SyncError);
        }

        if crc16_checksum(&data[..self.frame_size as usize]) != 0 {
            return Err(HcaError::ChecksumFailed);
        }

        // Decrypt only when the file uses a non-identity cipher.
        if self.ciph_type != 0 {
            cipher_decrypt(&self.cipher_table, &mut data[..self.frame_size as usize]);
        }

        // Re-initialize bitreader after decryption
        let mut br = BitReader::with_offset(data, 2); // Skip sync word

        // Unpack frame values
        let frame_acceptable_noise_level = br.read(9);
        let frame_evaluation_boundary = br.read(7);
        let packed_noise_level =
            (frame_acceptable_noise_level << 8).wrapping_sub(frame_evaluation_boundary);

        for ch in 0..self.channels as usize {
            self.unpack_scale_factors(ch, &mut br)?;
            self.unpack_intensity(ch, &mut br)?;
            self.calculate_resolution(ch, packed_noise_level);
            self.calculate_gain(ch);
        }

        for subframe in 0..HCA_SUBFRAMES {
            for ch in 0..self.channels as usize {
                self.dequantize_coefficients(ch, &mut br, subframe);
            }
        }

        Ok(br.position())
    }

    fn unpack_scale_factors(&mut self, ch: usize, br: &mut BitReader) -> Result<(), HcaError> {
        let channel = &mut self.channel[ch];
        let mut cs_count = channel.coded_count;
        let extra_count: usize;

        let delta_bits = br.read(3) as u8;

        if channel.channel_type == ChannelType::StereoSecondary
            || self.hfr_group_count == 0
            || self.version <= HCA_VERSION_200
        {
            extra_count = 0;
        } else {
            extra_count = self.hfr_group_count as usize;
            cs_count += extra_count;

            if cs_count > HCA_SAMPLES_PER_SUBFRAME {
                return Err(HcaError::UnpackError("invalid coded count".into()));
            }
        }

        Self::unpack_scale_factor_values(channel, br, cs_count, delta_bits)?;

        // Set derived HFR scales for v3.0
        for i in 0..extra_count {
            channel.scale_factors[HCA_SAMPLES_PER_SUBFRAME - 1 - i] =
                channel.scale_factors[cs_count - i];
        }

        Ok(())
    }

    fn unpack_scale_factor_values(
        channel: &mut StChannel,
        br: &mut BitReader,
        count: usize,
        delta_bits: u8,
    ) -> Result<(), HcaError> {
        if delta_bits == 0 {
            channel.scale_factors.fill(0);
            return Ok(());
        }

        let mut acc = br.acc();
        if delta_bits >= 6 {
            for entry in channel.scale_factors.iter_mut().take(count) {
                *entry = acc.read(6) as u8;
            }
            acc.sync(br);
            return Ok(());
        }

        let expected_delta = ((1 << delta_bits) - 1) as u8;
        let mut value = acc.read(6) as u8;
        channel.scale_factors[0] = value;
        for entry in channel.scale_factors.iter_mut().take(count).skip(1) {
            let delta = acc.read(delta_bits as u32) as u8;
            if delta == expected_delta {
                value = acc.read(6) as u8;
            } else {
                let candidate = value as i32 + delta as i32 - (expected_delta >> 1) as i32;
                if !(0..64).contains(&candidate) {
                    return Err(HcaError::UnpackError("invalid scalefactor".into()));
                }
                value = value.wrapping_sub(expected_delta >> 1).wrapping_add(delta) & 0x3f;
            }
            *entry = value;
        }
        acc.sync(br);
        Ok(())
    }

    fn unpack_intensity(&mut self, ch: usize, br: &mut BitReader) -> Result<(), HcaError> {
        let channel = &mut self.channel[ch];

        if channel.channel_type == ChannelType::StereoSecondary {
            if self.version <= HCA_VERSION_200 {
                Self::unpack_legacy_intensity(channel, br);
            } else {
                Self::unpack_delta_intensity(channel, br)?;
            }
        } else if self.version <= HCA_VERSION_200 {
            Self::unpack_hfr_scales(channel, br, self.hfr_group_count as usize);
        }

        Ok(())
    }

    fn unpack_legacy_intensity(channel: &mut StChannel, br: &mut BitReader) {
        let value = br.peek(4) as u8;
        channel.intensity[0] = value;
        if value >= 15 {
            return;
        }
        br.skip(4);
        for entry in channel.intensity.iter_mut().skip(1) {
            *entry = br.read(4) as u8;
        }
    }

    fn unpack_delta_intensity(channel: &mut StChannel, br: &mut BitReader) -> Result<(), HcaError> {
        let mut value = br.peek(4) as u8;
        if value >= 15 {
            br.skip(4);
            channel.intensity.fill(7);
            return Ok(());
        }

        br.skip(4);
        let delta_bits = br.read(2) as u8;
        channel.intensity[0] = value;
        if delta_bits == 3 {
            for entry in channel.intensity.iter_mut().skip(1) {
                *entry = br.read(4) as u8;
            }
            return Ok(());
        }

        let maximum_delta = ((2 << delta_bits) - 1) as u8;
        for entry in channel.intensity.iter_mut().skip(1) {
            let delta = br.read((delta_bits + 1) as usize) as u8;
            value = decode_intensity_delta(br, value, delta, maximum_delta)?;
            *entry = value;
        }
        Ok(())
    }

    fn unpack_hfr_scales(channel: &mut StChannel, br: &mut BitReader, count: usize) {
        let hfr_scales = &mut channel.scale_factors[HCA_SAMPLES_PER_SUBFRAME - count..];
        for entry in hfr_scales {
            *entry = br.read(6) as u8;
        }
    }

    fn calculate_resolution(&mut self, ch: usize, packed_noise_level: u32) {
        let channel = &mut self.channel[ch];
        let cr_count = channel.coded_count;
        let mut noise_count = 0usize;
        let mut valid_count = 0usize;

        for i in 0..cr_count {
            let scalefactor = channel.scale_factors[i];
            let new_resolution = resolution_for_scale_factor(
                scalefactor,
                self.ath_curve[i],
                packed_noise_level,
                i,
                self.min_resolution as u8,
                self.max_resolution as u8,
            );

            if scalefactor > 0 {
                if new_resolution == 0 {
                    channel.noises[noise_count] = i as u8;
                    noise_count += 1;
                } else {
                    channel.noises[HCA_SAMPLES_PER_SUBFRAME - 1 - valid_count] = i as u8;
                    valid_count += 1;
                }
            }
            channel.resolution[i] = new_resolution;
        }

        channel.noise_count = noise_count;
        channel.valid_count = valid_count;

        channel.resolution[cr_count..HCA_SAMPLES_PER_SUBFRAME].fill(0);
    }

    fn calculate_gain(&mut self, ch: usize) {
        let channel = &mut self.channel[ch];
        let cg_count = channel.coded_count;

        for i in 0..cg_count {
            let scalefactor_scale = DEQUANTIZER_SCALING_TABLE[channel.scale_factors[i] as usize];
            let resolution_scale = DEQUANTIZER_RANGE_TABLE[channel.resolution[i] as usize];
            channel.gain[i] = scalefactor_scale * resolution_scale;
        }
    }

    fn dequantize_coefficients(&mut self, ch: usize, br: &mut BitReader, subframe: usize) {
        let channel = &mut self.channel[ch];
        let cc_count = channel.coded_count;
        let spectra = &mut channel.spectra[subframe];
        let gain = &channel.gain;
        let resolution = &channel.resolution;

        // Register-resident, forward-only bit accumulator. Each coefficient
        // peeks a fixed-width code and consumes the *actual* prefix length, so
        // the bit position only ever moves forward. Keeping the next bits in a
        // u64 avoids the per-coefficient memory reload and second position
        // write that `read_hca_bits` + `advance_signed` incurred.
        let data = br.data();
        let len = data.len();
        let start = br.position();
        let mut byte_pos = start >> 3;

        // MSB-first accumulator: valid bits live in the high end, low (unfilled)
        // bits are zero, which reproduces the zero-padding-past-EOF semantics.
        let mut acc: u64 = 0;
        let mut nbits: u32 = 0;
        macro_rules! refill {
            () => {
                while nbits <= 56 && byte_pos < len {
                    acc |= (data[byte_pos] as u64) << (56 - nbits);
                    nbits += 8;
                    byte_pos += 1;
                }
            };
        }

        refill!();
        // Drop the already-consumed bits in the leading byte.
        let lead = (start & 7) as u32;
        acc <<= lead;
        nbits = nbits.saturating_sub(lead);

        // Bits consumed relative to the byte-aligned base `(start >> 3) * 8`.
        let mut consumed: usize = lead as usize;

        for i in 0..cc_count {
            if nbits < 32 {
                refill!();
            }

            let res = resolution[i];
            let bits = MAX_BIT_TABLE[res as usize] as u32;
            // Peek the fixed-width code; for resolution 0, bits == 0.
            let code = if bits == 0 {
                0
            } else {
                (acc >> (64 - bits)) as u32
            };

            // `used` is the actual prefix length consumed, computed
            // arithmetically (see DEQ_USED_BASE) instead of via a table load.
            // This keeps the loop-carried chain (acc -> code -> used -> acc)
            // in registers; the value lookup stays a load but is off that
            // chain.
            let used = DEQ_USED_BASE[res as usize] as u32
                + (code >= DEQ_USED_THRESH[res as usize] as u32) as u32;

            let qc: f32 = if res <= 7 {
                READ_VAL_TABLE[((res as usize) << 4) + code as usize]
            } else {
                // Sign-magnitude: bit 0 is sign, bits 1+ are magnitude.
                ((1 - ((code & 1) << 1) as i32) * (code >> 1) as i32) as f32
            };

            acc <<= used;
            nbits = nbits.saturating_sub(used);
            consumed += used as usize;

            spectra[i] = gain[i] * qc;
        }

        br.set_position(((start >> 3) << 3) + consumed);

        // Clean rest of spectra
        spectra[cc_count..HCA_SAMPLES_PER_SUBFRAME].fill(0.0);
    }

    fn decode_block_transform(&mut self) {
        if self.min_resolution > 0 && self.bands_per_hfr_group == 0 && self.stereo_band_count == 0 {
            for subframe in 0..HCA_SUBFRAMES {
                self.transform_channels(subframe);
            }
            return;
        }

        for subframe in 0..HCA_SUBFRAMES {
            self.restore_transform_bands(subframe);
            self.transform_channels(subframe);
        }
    }

    fn restore_transform_bands(&mut self, subframe: usize) {
        for channel in 0..self.channels as usize {
            self.reconstruct_noise(channel, subframe);
            self.reconstruct_high_frequency(channel, subframe);
        }
        if self.stereo_band_count == 0 {
            return;
        }
        for channel in 0..(self.channels as usize).saturating_sub(1) {
            self.apply_intensity_stereo(channel, subframe);
            self.apply_ms_stereo(channel, subframe);
        }
    }

    fn transform_channels(&mut self, subframe: usize) {
        for channel in self.channel.iter_mut().take(self.channels as usize) {
            imdct_transform(channel, subframe);
        }
    }

    fn reconstruct_noise(&mut self, ch: usize, subframe: usize) {
        if self.min_resolution > 0 {
            return;
        }
        let channel = &self.channel[ch];
        if channel.valid_count == 0 || channel.noise_count == 0 {
            return;
        }
        if !(self.ms_stereo == 0 || channel.channel_type == ChannelType::StereoPrimary) {
            return;
        }

        let mut r = self.random;

        for i in 0..self.channel[ch].noise_count {
            r = r.wrapping_mul(0x343FD).wrapping_add(0x269EC3);

            let random_index = HCA_SAMPLES_PER_SUBFRAME - self.channel[ch].valid_count
                + (((r & 0x7FFF) as usize * self.channel[ch].valid_count) >> 15);

            let noise_index = self.channel[ch].noises[i] as usize;
            let valid_index = self.channel[ch].noises[random_index] as usize;

            let sf_noise = self.channel[ch].scale_factors[noise_index];
            let sf_valid = self.channel[ch].scale_factors[valid_index];
            let sc_index = (sf_noise as i32 - sf_valid as i32 + 62).max(0) as usize;

            let spectra_valid = self.channel[ch].spectra[subframe][valid_index];
            self.channel[ch].spectra[subframe][noise_index] =
                SCALE_CONVERSION_TABLE[sc_index] * spectra_valid;
        }

        self.random = r;
    }

    fn reconstruct_high_frequency(&mut self, ch: usize, subframe: usize) {
        if self.bands_per_hfr_group == 0 {
            return;
        }
        if self.channel[ch].channel_type == ChannelType::StereoSecondary {
            return;
        }

        let start_band = (self.stereo_band_count + self.base_band_count) as usize;
        let mut highband = start_band;
        // Use i32 to match C's signed int lowband, which allows < 0 check
        let mut lowband = start_band as i32 - 1;

        // In C, hfr_group_count is unsigned, so (hfr_group_count >= 0) is always true
        // meaning group_limit = hfr_group_count (v2.0) or hfr_group_count >> 1 (v3.0)
        let group_limit = if self.version <= HCA_VERSION_200 {
            self.hfr_group_count as usize
        } else {
            (self.hfr_group_count as usize) >> 1
        };

        for group in 0..self.hfr_group_count as usize {
            let lowband_sub: i32 = if group >= group_limit { 0 } else { 1 };

            for _ in 0..self.bands_per_hfr_group as usize {
                if highband >= self.total_band_count as usize || lowband < 0 {
                    break;
                }

                let hfr_scale =
                    self.channel[ch].scale_factors[128 - self.hfr_group_count as usize + group];
                let sf_low = self.channel[ch].scale_factors[lowband as usize];
                let sc_index = (hfr_scale as i32 - sf_low as i32 + 63).max(0) as usize;

                let spectra_low = self.channel[ch].spectra[subframe][lowband as usize];
                self.channel[ch].spectra[subframe][highband] =
                    SCALE_CONVERSION_TABLE[sc_index] * spectra_low;

                highband += 1;
                lowband -= lowband_sub;
            }
        }

        if highband > 0 {
            self.channel[ch].spectra[subframe][highband - 1] = 0.0;
        }
    }

    fn apply_intensity_stereo(&mut self, ch: usize, subframe: usize) {
        if self.channel[ch].channel_type != ChannelType::StereoPrimary {
            return;
        }

        let ratio_l = INTENSITY_RATIO_TABLE[self.channel[ch + 1].intensity[subframe] as usize];
        let ratio_r = 2.0 - ratio_l;

        for band in self.base_band_count as usize..self.total_band_count as usize {
            let coef = self.channel[ch].spectra[subframe][band];
            self.channel[ch].spectra[subframe][band] = coef * ratio_l;
            self.channel[ch + 1].spectra[subframe][band] = coef * ratio_r;
        }
    }

    fn apply_ms_stereo(&mut self, ch: usize, subframe: usize) {
        if self.ms_stereo == 0 {
            return;
        }
        if self.channel[ch].channel_type != ChannelType::StereoPrimary {
            return;
        }

        const RATIO: f32 = 0.707_106_77;

        for band in self.base_band_count as usize..self.total_band_count as usize {
            let l = self.channel[ch].spectra[subframe][band];
            let r = self.channel[ch + 1].spectra[subframe][band];
            self.channel[ch].spectra[subframe][band] = (l + r) * RATIO;
            self.channel[ch + 1].spectra[subframe][band] = (l - r) * RATIO;
        }
    }
}

#[inline]
pub(crate) fn pcm_f32_to_i16(sample: f32) -> i16 {
    const SCALE_F: f32 = 32768.0;
    let scaled = (sample * SCALE_F) as i32;
    scaled.clamp(-32768, 32767) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc16_checksum() {
        assert_eq!(crc16_checksum(&[]), 0);
        // Known CRC16 for a simple byte sequence
        let result = crc16_checksum(&[0x48, 0x43, 0x41, 0x00]);
        assert_ne!(result, 0, "CRC of non-empty data should not be 0");
    }

    #[test]
    fn test_header_ceil2() {
        assert_eq!(header_ceil2(10, 3), 4);
        assert_eq!(header_ceil2(9, 3), 3);
        assert_eq!(header_ceil2(0, 3), 0);
        assert_eq!(header_ceil2(10, 0), 0);
        assert_eq!(header_ceil2(1, 1), 1);
        assert_eq!(header_ceil2(7, 2), 4);
    }

    #[test]
    fn test_is_hca_file_valid() {
        // HCA\0 signature (0x48434100) + version (0x0102) + header_size (0x0060)
        let data = [0x48, 0x43, 0x41, 0x00, 0x01, 0x02, 0x00, 0x60];
        let result = ClHca::is_hca_file(&data);
        assert_eq!(result, Some(0x0060));
    }

    #[test]
    fn test_is_hca_file_invalid() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(ClHca::is_hca_file(&data), None);

        let short = [0x48, 0x43];
        assert_eq!(ClHca::is_hca_file(&short), None);
    }

    #[test]
    fn test_clhca_new_defaults() {
        let hca = ClHca::new();
        assert!(!hca.is_valid);
        assert_eq!(hca.version, 0);
        assert_eq!(hca.channels, 0);
        assert_eq!(hca.random, HCA_DEFAULT_RANDOM);
        assert_eq!(hca.rva_volume, 1.0);
    }

    #[test]
    fn test_refactored_decoder_helpers() {
        let mut channel_types = [ChannelType::Discrete; 8];
        assign_stereo_channel_types(&mut channel_types[..4], 4, 0);
        assert_eq!(channel_types[0], ChannelType::StereoPrimary);
        assert_eq!(channel_types[1], ChannelType::StereoSecondary);
        assert_eq!(channel_types[2], ChannelType::StereoPrimary);
        assert_eq!(channel_types[3], ChannelType::StereoSecondary);
        assign_stereo_channel_types(&mut channel_types[..5], 5, 2);
        assert_eq!(channel_types[3], ChannelType::StereoPrimary);
        assign_stereo_channel_types(&mut channel_types[..6], 6, 0);
        assert_eq!(channel_types[4], ChannelType::StereoPrimary);
        assign_stereo_channel_types(&mut channel_types, 8, 0);
        assert_eq!(channel_types[6], ChannelType::StereoPrimary);

        assert_eq!(sample_quality(1.5, 32768.0), (1, 0));
        assert_eq!(sample_quality(0.0, 32768.0), (0, 1));
        assert_eq!(sample_quality(0.5, 32768.0), (0, 0));
        assert_eq!(resolution_for_scale_factor(0, 0, 0, 0, 0, 15), 0);
        assert_eq!(resolution_for_scale_factor(63, 0, 0, 0, 1, 15), 15);
        assert_eq!(resolution_for_scale_factor(1, 100, 0, 0, 0, 15), 0);

        let mut hca = Box::new(ClHca::new());
        hca.channels = 1;
        hca.channel[0].wave[0][0] = 0.5;
        let mut mono = vec![0; HCA_SAMPLES_PER_FRAME];
        hca.read_samples_16(&mut mono);
        assert_eq!(mono[0], pcm_f32_to_i16(0.5));

        hca.channels = 3;
        hca.channel[0].wave[0][0] = 0.25;
        hca.channel[1].wave[0][0] = 0.5;
        hca.channel[2].wave[0][0] = 0.75;
        let mut interleaved = vec![0; HCA_SAMPLES_PER_FRAME * 3];
        hca.read_samples_16(&mut interleaved);
        assert_eq!(interleaved[0], pcm_f32_to_i16(0.25));
        assert_eq!(interleaved[1], pcm_f32_to_i16(0.5));
        assert_eq!(interleaved[2], pcm_f32_to_i16(0.75));
    }

    #[test]
    fn test_scale_factor_and_intensity_unpack_helpers() {
        let mut channel = StChannel::default();
        channel.scale_factors.fill(9);
        ClHca::unpack_scale_factor_values(&mut channel, &mut BitReader::new(&[]), 2, 0).unwrap();
        assert!(channel.scale_factors.iter().all(|&value| value == 0));

        let fixed_data = [0b1010_1001, 0b0101_0000];
        ClHca::unpack_scale_factor_values(&mut channel, &mut BitReader::new(&fixed_data), 2, 6)
            .unwrap();
        assert_eq!(&channel.scale_factors[..2], &[42, 21]);

        let mut escaped_reader = BitReader::new(&[0b1010_0000]);
        assert_eq!(
            decode_intensity_delta(&mut escaped_reader, 0, 3, 3).unwrap(),
            10
        );
        assert!(decode_intensity_delta(&mut BitReader::new(&[]), 15, 2, 3).is_err());

        let mut channel = StChannel::default();
        ClHca::unpack_delta_intensity(&mut channel, &mut BitReader::new(&[0xf0])).unwrap();
        assert!(channel.intensity.iter().all(|&value| value == 7));

        ClHca::unpack_delta_intensity(&mut channel, &mut BitReader::new(&[0x20, 0x00])).unwrap();
        assert!(channel.intensity.iter().all(|&value| value == 2));

        let mut writer = crate::hca::bitreader::BitWriter::new(8);
        writer.write(2, 4);
        writer.write(3, 2);
        for value in 1..HCA_SUBFRAMES as u32 {
            writer.write(value, 4);
        }
        ClHca::unpack_delta_intensity(&mut channel, &mut BitReader::new(writer.data())).unwrap();
        assert_eq!(channel.intensity, [2, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_deq_used_matches_read_bit_table() {
        use crate::hca::tables::READ_BIT_TABLE;

        // Prefix resolutions (0..=7): the arithmetic model must reproduce
        // READ_BIT_TABLE for every fixed-width code.
        for res in 0..=7usize {
            let bits = MAX_BIT_TABLE[res] as u32;
            for code in 0..(1u32 << bits) {
                let expect = READ_BIT_TABLE[(res << 4) + code as usize] as u32;
                let got = DEQ_USED_BASE[res] as u32 + (code >= DEQ_USED_THRESH[res] as u32) as u32;
                assert_eq!(got, expect, "res {res} code {code}");
            }
        }
        // Sign-magnitude resolutions (8..=15): bits-1 when magnitude is 0.
        for res in 8..16usize {
            let bits = MAX_BIT_TABLE[res] as u32;
            for code in 0..(1u32 << bits) {
                let expect = if code >> 1 == 0 { bits - 1 } else { bits };
                let got = DEQ_USED_BASE[res] as u32 + (code >= DEQ_USED_THRESH[res] as u32) as u32;
                assert_eq!(got, expect, "res {res} code {code}");
            }
        }
    }

    #[test]
    fn test_clhca_clear() {
        let mut hca = ClHca::new();
        hca.version = 0x0102;
        hca.channels = 2;
        hca.clear();
        assert_eq!(hca.version, 0);
        assert_eq!(hca.channels, 0);
    }
}
