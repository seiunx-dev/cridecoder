//! HCA encoder - encodes PCM audio to HCA format.
//!
//! The frame encoder follows the VGAudio/PyCriCodecs flow: MDCT, scale factor
//! selection, bit-budget driven noise/resolution search, quantization, and CRI
//! bit packing.

use super::bitreader::BitWriter;
use super::cipher::cipher_init;
use super::decoder::{
    assign_stereo_channel_types, crc16_checksum, ChannelType, HCA_MAX_CHANNELS,
    HCA_SAMPLES_PER_FRAME, HCA_SAMPLES_PER_SUBFRAME, HCA_SUBFRAMES, HCA_VERSION_200,
};
use super::tables::{DEQUANTIZER_SCALING_TABLE, IMDCT_WINDOW, MAX_BIT_TABLE};
use std::f32::consts::{PI, SQRT_2};
use std::io::{self, Seek, Write};
use std::sync::LazyLock;
use thiserror::Error;

const HEADER_SIZE: usize = 0x60;
const ENCODER_DELAY: u32 = HCA_SAMPLES_PER_SUBFRAME as u32;
const DEFAULT_TRACK_COUNT: u32 = 1;
const MIN_RESOLUTION: u8 = 1;
const MAX_RESOLUTION: u8 = 15;

const DEFAULT_CHANNEL_MAPPING: [u8; 9] = [0, 1, 0, 4, 0, 1, 3, 7, 3];
const VALID_CHANNEL_MAPPINGS: [[u8; 8]; 8] = [
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0],
];

const SCALE_TO_RESOLUTION_CURVE: [u8; 59] = [
    0x0F, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0C, 0x0C, 0x0C,
    0x0C, 0x0C, 0x0C, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
    0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x07, 0x06, 0x06, 0x05,
    0x04, 0x04, 0x04, 0x03, 0x03, 0x03, 0x02, 0x02, 0x02, 0x02, 0x01,
];

const QUANTIZER_INVERSE_STEP_SIZE: [f32; 16] = [
    0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 15.5, 31.5, 63.5, 127.5, 255.5, 511.5, 1023.5, 2047.5,
];

const QUANTIZE_SPECTRUM_BITS: [[u8; 16]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 3, 3, 2, 3, 3, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 3, 3, 3, 3, 3, 3, 3, 4, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 3, 3, 3, 3, 3, 4, 4, 4, 0, 0],
    [0, 0, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 0],
    [0, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4],
];

const QUANTIZE_SPECTRUM_VALUE: [[u32; 16]; 8] = [
    [
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    ],
    [
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x3, 0x0, 0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    ],
    [
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x7, 0x2, 0x0, 0x1, 0x6, 0x0, 0x0, 0x0, 0x0, 0x0,
    ],
    [
        0x0, 0x0, 0x0, 0x0, 0x0, 0x7, 0x5, 0x3, 0x0, 0x2, 0x4, 0x6, 0x0, 0x0, 0x0, 0x0,
    ],
    [
        0x0, 0x0, 0x0, 0x0, 0xF, 0x6, 0x4, 0x2, 0x0, 0x1, 0x3, 0x5, 0xE, 0x0, 0x0, 0x0,
    ],
    [
        0x0, 0x0, 0x0, 0xF, 0xD, 0xB, 0x4, 0x2, 0x0, 0x1, 0x3, 0xA, 0xC, 0xE, 0x0, 0x0,
    ],
    [
        0x0, 0x0, 0xF, 0xD, 0xB, 0x9, 0x7, 0x2, 0x0, 0x1, 0x6, 0x8, 0xA, 0xC, 0xE, 0x0,
    ],
    [
        0x0, 0xF, 0xD, 0xB, 0x9, 0x7, 0x5, 0x3, 0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE,
    ],
];

static QUANTIZER_DEAD_ZONE: LazyLock<[f32; 16]> = LazyLock::new(|| {
    let hex: [u32; 16] = [
        0x00000000, 0x3EAAAAAB, 0x3E4CCCCD, 0x3E124925, 0x3DE38E39, 0x3DBA2E8C, 0x3D9D89D9,
        0x3D888889, 0x3D042108, 0x3C820821, 0x3C010204, 0x3B808081, 0x3B004020, 0x3A802008,
        0x3A001002, 0x39800801,
    ];
    let mut result = [0.0f32; 16];
    for i in 0..16 {
        result[i] = f32::from_bits(hex[i]);
    }
    result
});

static INTENSITY_RATIO_BOUNDS: LazyLock<[f32; 14]> = LazyLock::new(|| {
    let hex: [u32; 14] = [
        0x3FF6DB6E, 0x3FE49249, 0x3FD24925, 0x3FC00000, 0x3FADB6DB, 0x3F9B6DB7, 0x3F892492,
        0x3F6DB6DB, 0x3F492492, 0x3F249249, 0x3F000000, 0x3EB6DB6E, 0x3E5B6DB7, 0x3D924925,
    ];
    let mut result = [0.0f32; 14];
    for i in 0..14 {
        result[i] = f32::from_bits(hex[i]);
    }
    result
});

#[derive(Debug, Error)]
pub enum HcaEncoderError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),
    #[error("Invalid channel count: {0}")]
    InvalidChannelCount(u32),
    #[error("Frame too large")]
    FrameTooLarge,
    #[error("No samples provided")]
    NoSamples,
}

/// HCA encoder configuration.
#[derive(Debug, Clone)]
pub struct HcaEncoderConfig {
    pub sample_rate: u32,
    pub channels: u32,
    pub bitrate: u32,
    pub encryption_key: Option<u64>,
    pub loop_start: Option<u32>,
    pub loop_end: Option<u32>,
}

impl Default for HcaEncoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            bitrate: 256000,
            encryption_key: None,
            loop_start: None,
            loop_end: None,
        }
    }
}

impl HcaEncoderConfig {
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        Self {
            sample_rate,
            channels,
            ..Default::default()
        }
    }

    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.bitrate = bitrate;
        self
    }

    pub fn with_encryption(mut self, key: u64) -> Self {
        self.encryption_key = Some(key);
        self
    }
}

#[derive(Clone)]
struct ChannelEncodeState {
    channel_type: ChannelType,
    coded_count: usize,
    wave: [[f32; HCA_SAMPLES_PER_SUBFRAME]; HCA_SUBFRAMES],
    spectra: [[f32; HCA_SAMPLES_PER_SUBFRAME]; HCA_SUBFRAMES],
    scaled_spectra: [[f32; HCA_SUBFRAMES]; HCA_SAMPLES_PER_SUBFRAME],
    quantized_spectra: [[i32; HCA_SAMPLES_PER_SUBFRAME]; HCA_SUBFRAMES],
    scale_factors: [u8; HCA_SAMPLES_PER_SUBFRAME],
    resolutions: [u8; HCA_SAMPLES_PER_SUBFRAME],
    intensity: [u8; HCA_SUBFRAMES],
    hfr_group_average_spectra: [f32; HCA_SUBFRAMES],
    hfr_scales: [u8; HCA_SUBFRAMES],
    scale_factor_delta_bits: u8,
    header_length_bits: usize,
    mdct_previous: [f32; HCA_SAMPLES_PER_SUBFRAME],
}

impl Default for ChannelEncodeState {
    fn default() -> Self {
        Self {
            channel_type: ChannelType::Discrete,
            coded_count: 0,
            wave: [[0.0; HCA_SAMPLES_PER_SUBFRAME]; HCA_SUBFRAMES],
            spectra: [[0.0; HCA_SAMPLES_PER_SUBFRAME]; HCA_SUBFRAMES],
            scaled_spectra: [[0.0; HCA_SUBFRAMES]; HCA_SAMPLES_PER_SUBFRAME],
            quantized_spectra: [[0; HCA_SAMPLES_PER_SUBFRAME]; HCA_SUBFRAMES],
            scale_factors: [0; HCA_SAMPLES_PER_SUBFRAME],
            resolutions: [0; HCA_SAMPLES_PER_SUBFRAME],
            intensity: [0; HCA_SUBFRAMES],
            hfr_group_average_spectra: [0.0; HCA_SUBFRAMES],
            hfr_scales: [0; HCA_SUBFRAMES],
            scale_factor_delta_bits: 0,
            header_length_bits: 0,
            mdct_previous: [0.0; HCA_SAMPLES_PER_SUBFRAME],
        }
    }
}

fn quantize_intensity(left: f32, right: f32, total: f32) -> (u8, f32) {
    let combined = left + right;
    if combined <= 0.0 {
        return (0, 1.0);
    }

    let stored_value = 2.0 * left / combined;
    let energy_ratio = if total > 0.0 {
        (combined / total).clamp(0.5, SQRT_2 / 2.0)
    } else {
        1.0
    };
    let mut value = 1;
    while value < 13 && INTENSITY_RATIO_BOUNDS[value] >= stored_value {
        value += 1;
    }
    (value as u8, energy_ratio)
}

fn hfr_spectra_average(
    channel: &ChannelEncodeState,
    mut band: usize,
    group_size: usize,
) -> (f32, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for _ in 0..group_size {
        if band >= HCA_SAMPLES_PER_SUBFRAME {
            break;
        }
        for subframe in 0..HCA_SUBFRAMES {
            sum += channel.spectra[subframe][band].abs();
        }
        count += HCA_SUBFRAMES;
        band += 1;
    }
    let average = if count == 0 { 0.0 } else { sum / count as f32 };
    (average, band)
}

fn hfr_source_average(
    channel: &ChannelEncodeState,
    hfr_start_band: usize,
    hfr_band_count: usize,
    mut band: usize,
    group_size: usize,
) -> (f32, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for _ in 0..group_size {
        if band >= hfr_band_count || band >= hfr_start_band {
            break;
        }
        let source_band = hfr_start_band - band - 1;
        for subframe in 0..HCA_SUBFRAMES {
            sum += channel.scaled_spectra[source_band][subframe].abs();
        }
        count += HCA_SUBFRAMES;
        band += 1;
    }
    let average = if count == 0 { 0.0 } else { sum / count as f32 };
    (average, band)
}

fn spectral_bit_count(channel: &ChannelEncodeState, band: usize, resolution: usize) -> usize {
    if resolution >= 8 {
        let base_bits = MAX_BIT_TABLE[resolution] as usize - 1;
        let dead_zone = QUANTIZER_DEAD_ZONE[resolution];
        return channel.scaled_spectra[band]
            .iter()
            .map(|value| base_bits + usize::from(value.abs() >= dead_zone))
            .sum();
    }

    let step_size = QUANTIZER_INVERSE_STEP_SIZE[resolution];
    let shift_up = step_size + 1.0;
    let shift_down = (step_size + 0.5 - 8.0) as i32;
    channel.scaled_spectra[band]
        .iter()
        .map(|value| {
            let quantized = (*value * step_size + shift_up) as i32 - shift_down;
            QUANTIZE_SPECTRUM_BITS[resolution][quantized.clamp(0, 15) as usize] as usize
        })
        .sum()
}

/// HCA encoder.
pub struct HcaEncoder {
    config: HcaEncoderConfig,
    frame_size: u32,
    total_band_count: u32,
    base_band_count: u32,
    stereo_band_count: u32,
    hfr_band_count: u32,
    bands_per_hfr_group: u32,
    hfr_group_count: u32,
    track_count: u32,
    channel_config: u32,
    encoder_delay: u32,
    encoder_padding: u32,
    channel: Vec<ChannelEncodeState>,
    encrypt_table: [u8; 256],
}

impl HcaEncoder {
    pub fn new(config: HcaEncoderConfig) -> Result<Self, HcaEncoderError> {
        if config.sample_rate == 0 || config.sample_rate > 0x7FFFFF {
            return Err(HcaEncoderError::InvalidSampleRate(config.sample_rate));
        }
        if config.channels == 0 || config.channels > HCA_MAX_CHANNELS as u32 {
            return Err(HcaEncoderError::InvalidChannelCount(config.channels));
        }

        let frame_size = Self::calculate_frame_size(config.bitrate, config.sample_rate);
        let (total_band_count, base_band_count, stereo_band_count, bands_per_hfr_group) =
            Self::calculate_band_counts(&config, frame_size);
        let hfr_band_count = total_band_count - base_band_count - stereo_band_count;
        let hfr_group_count = ceil_div(hfr_band_count, bands_per_hfr_group);
        let track_count = DEFAULT_TRACK_COUNT;
        let channel_config = Self::default_channel_config(config.channels, track_count)?;

        let mut cipher_table = [0u8; 256];
        if let Some(key) = config.encryption_key {
            cipher_init(&mut cipher_table, 56, key);
        } else {
            cipher_init(&mut cipher_table, 0, 0);
        }

        let mut encrypt_table = [0u8; 256];
        for (plain, &encrypted) in cipher_table.iter().enumerate() {
            encrypt_table[encrypted as usize] = plain as u8;
        }

        let mut encoder = Self {
            config,
            frame_size,
            total_band_count,
            base_band_count,
            stereo_band_count,
            hfr_band_count,
            bands_per_hfr_group,
            hfr_group_count,
            track_count,
            channel_config,
            encoder_delay: ENCODER_DELAY,
            encoder_padding: 0,
            channel: vec![ChannelEncodeState::default(); HCA_MAX_CHANNELS],
            encrypt_table,
        };
        encoder.set_channel_types();
        Ok(encoder)
    }

    fn calculate_frame_size(bitrate: u32, sample_rate: u32) -> u32 {
        let size = (bitrate as u64 * HCA_SAMPLES_PER_FRAME as u64) / (sample_rate as u64 * 8);
        (size as u32).clamp(0x8, 0xFFFF)
    }

    fn calculate_band_counts(config: &HcaEncoderConfig, _frame_size: u32) -> (u32, u32, u32, u32) {
        let mut cutoff_frequency = config.sample_rate / 2;
        let bitrate = config.bitrate.max(1);
        let pcm_bitrate = config.sample_rate * config.channels * 16;
        let (hfr_ratio, cutoff_ratio) = if config.channels <= 1 || pcm_bitrate / bitrate <= 6 {
            (6, 12)
        } else {
            (8, 16)
        };

        if bitrate < pcm_bitrate / cutoff_ratio {
            cutoff_frequency =
                cutoff_frequency.min(cutoff_ratio * bitrate / (32 * config.channels));
        }

        let total_band_count = ((cutoff_frequency as f64 * 256.0 / config.sample_rate as f64)
            .round() as u32)
            .clamp(1, HCA_SAMPLES_PER_SUBFRAME as u32);
        let hfr_start_band = total_band_count
            .min(((hfr_ratio as f64 * bitrate as f64 * 128.0) / pcm_bitrate as f64).round() as u32);
        let stereo_start_band = if hfr_ratio == 6 {
            hfr_start_band
        } else {
            hfr_start_band.div_ceil(2)
        };

        let hfr_band_count = total_band_count - hfr_start_band;
        let bands_per_hfr_group = hfr_band_count.div_ceil(8);

        (
            total_band_count,
            stereo_start_band,
            hfr_start_band - stereo_start_band,
            bands_per_hfr_group,
        )
    }

    fn default_channel_config(channels: u32, track_count: u32) -> Result<u32, HcaEncoderError> {
        let channels_per_track = channels / track_count;
        let channel_config = DEFAULT_CHANNEL_MAPPING
            .get(channels_per_track as usize)
            .copied()
            .unwrap_or(0) as u32;
        let valid = VALID_CHANNEL_MAPPINGS
            .get((channels_per_track - 1) as usize)
            .and_then(|row| row.get(channel_config as usize))
            .copied()
            .unwrap_or(0);
        if valid == 0 {
            Err(HcaEncoderError::InvalidChannelCount(channels))
        } else {
            Ok(channel_config)
        }
    }

    fn set_channel_types(&mut self) {
        let mut types = vec![ChannelType::Discrete; self.config.channels as usize];
        let channels_per_track = self.config.channels / self.track_count;

        if self.stereo_band_count > 0 && channels_per_track > 1 {
            for track in 0..self.track_count as usize {
                let base = track * channels_per_track as usize;
                let end = base + channels_per_track as usize;
                assign_stereo_channel_types(
                    &mut types[base..end],
                    channels_per_track as usize,
                    self.channel_config,
                );
            }
        }

        for (i, channel_type) in types.iter().enumerate() {
            self.channel[i].channel_type = *channel_type;
            self.channel[i].coded_count = if *channel_type == ChannelType::StereoSecondary {
                self.base_band_count as usize
            } else {
                (self.base_band_count + self.stereo_band_count) as usize
            };
        }
    }

    /// Encode interleaved f32 PCM samples to HCA.
    pub fn encode<W: Write + Seek>(
        &mut self,
        samples: &[f32],
        writer: &mut W,
    ) -> Result<(), HcaEncoderError> {
        if samples.is_empty() {
            return Err(HcaEncoderError::NoSamples);
        }

        for ch in self.channel.iter_mut().take(self.config.channels as usize) {
            ch.mdct_previous.fill(0.0);
        }

        let channels = self.config.channels as usize;
        let sample_frames = samples.len().div_ceil(channels);
        let frame_count =
            (sample_frames as u32 + self.encoder_delay).div_ceil(HCA_SAMPLES_PER_FRAME as u32);
        self.encoder_padding =
            frame_count * HCA_SAMPLES_PER_FRAME as u32 - self.encoder_delay - sample_frames as u32;

        self.write_header(writer, frame_count)?;

        let frame_samples = HCA_SAMPLES_PER_FRAME * channels;
        for frame_idx in 0..frame_count as usize {
            let start = frame_idx * frame_samples;
            let end = (start + frame_samples).min(samples.len());

            let mut frame_data = vec![0.0f32; frame_samples];
            if start < samples.len() {
                frame_data[..end - start].copy_from_slice(&samples[start..end]);
            }

            self.encode_frame(writer, &frame_data)?;
        }

        Ok(())
    }

    fn write_header<W: Write + Seek>(
        &self,
        writer: &mut W,
        frame_count: u32,
    ) -> Result<(), HcaEncoderError> {
        let mut header = vec![0u8; HEADER_SIZE];

        header[0..4].copy_from_slice(b"HCA\0");
        write_be_u16(&mut header, 4, HCA_VERSION_200 as u16);
        write_be_u16(&mut header, 6, HEADER_SIZE as u16);

        header[8..12].copy_from_slice(b"fmt\0");
        header[12] = self.config.channels as u8;
        header[13..16].copy_from_slice(&(self.config.sample_rate & 0x00FF_FFFF).to_be_bytes()[1..]);
        write_be_u32(&mut header, 16, frame_count);
        write_be_u16(&mut header, 20, self.encoder_delay as u16);
        write_be_u16(&mut header, 22, self.encoder_padding as u16);

        header[24..28].copy_from_slice(b"comp");
        write_be_u16(&mut header, 28, self.frame_size as u16);
        header[30] = MIN_RESOLUTION;
        header[31] = MAX_RESOLUTION;
        header[32] = self.track_count as u8;
        header[33] = self.channel_config as u8;
        header[34] = self.total_band_count as u8;
        header[35] = self.base_band_count as u8;
        header[36] = self.stereo_band_count as u8;
        header[37] = self.bands_per_hfr_group as u8;

        let mut position = 40usize;
        if let (Some(loop_start), Some(loop_end)) = (self.config.loop_start, self.config.loop_end) {
            let loop_start = loop_start + self.encoder_delay;
            let loop_end = loop_end + self.encoder_delay;
            let loop_start_frame = loop_start / HCA_SAMPLES_PER_FRAME as u32;
            let loop_start_delay = loop_start % HCA_SAMPLES_PER_FRAME as u32;
            let mut loop_end_frame = loop_end / HCA_SAMPLES_PER_FRAME as u32;
            let mut loop_end_padding =
                HCA_SAMPLES_PER_FRAME as u32 - (loop_end % HCA_SAMPLES_PER_FRAME as u32);
            if loop_end_padding == HCA_SAMPLES_PER_FRAME as u32 {
                loop_end_frame = loop_end_frame.saturating_sub(1);
                loop_end_padding = 0;
            }

            header[position..position + 4].copy_from_slice(b"loop");
            write_be_u32(&mut header, position + 4, loop_start_frame);
            write_be_u32(&mut header, position + 8, loop_end_frame);
            write_be_u16(&mut header, position + 12, loop_start_delay as u16);
            write_be_u16(&mut header, position + 14, loop_end_padding as u16);
            position += 16;
        }

        header[position..position + 4].copy_from_slice(b"ciph");
        write_be_u16(
            &mut header,
            position + 4,
            if self.config.encryption_key.is_some() {
                56
            } else {
                0
            },
        );
        position += 6;

        header[position..position + 4].copy_from_slice(b"pad\0");

        if self.config.encryption_key.is_some() {
            mask_header(&mut header);
        }

        let crc = crc16_checksum(&header[..HEADER_SIZE - 2]);
        write_be_u16(&mut header, HEADER_SIZE - 2, crc);

        writer.write_all(&header)?;
        Ok(())
    }

    fn encode_frame<W: Write>(
        &mut self,
        writer: &mut W,
        samples: &[f32],
    ) -> Result<(), HcaEncoderError> {
        self.pcm_to_float(samples);
        self.run_mdct();
        self.encode_intensity_stereo();
        self.calculate_scale_factors();
        self.scale_spectra();
        self.calculate_hfr_group_averages();
        self.calculate_hfr_scale();
        self.calculate_frame_header_length();
        let acceptable_noise_level = self.calculate_noise_level()?;
        let evaluation_boundary = self.calculate_evaluation_boundary(acceptable_noise_level)?;
        self.calculate_frame_resolutions(acceptable_noise_level, evaluation_boundary);
        self.quantize_spectra();

        let mut frame = self.pack_frame(acceptable_noise_level, evaluation_boundary)?;
        if self.config.encryption_key.is_some() {
            for byte in &mut frame {
                *byte = self.encrypt_table[*byte as usize];
            }
        }

        let checksum = crc16_checksum(&frame[..self.frame_size as usize - 2]);
        frame[self.frame_size as usize - 2] = (checksum >> 8) as u8;
        frame[self.frame_size as usize - 1] = (checksum & 0xFF) as u8;

        writer.write_all(&frame)?;
        Ok(())
    }

    fn pcm_to_float(&mut self, samples: &[f32]) {
        let channels = self.config.channels as usize;
        for c in 0..channels {
            for sf in 0..HCA_SUBFRAMES {
                for i in 0..HCA_SAMPLES_PER_SUBFRAME {
                    let frame_index = sf * HCA_SAMPLES_PER_SUBFRAME + i;
                    let sample_index = frame_index * channels + c;
                    self.channel[c].wave[sf][i] = samples
                        .get(sample_index)
                        .copied()
                        .unwrap_or(0.0)
                        .clamp(-1.0, 1.0);
                }
            }
        }
    }

    fn run_mdct(&mut self) {
        for c in 0..self.config.channels as usize {
            for sf in 0..HCA_SUBFRAMES {
                mdct_transform(&mut self.channel[c], sf);
            }
        }
    }

    fn encode_intensity_stereo(&mut self) {
        if self.stereo_band_count == 0 {
            return;
        }

        for c in 0..self.config.channels as usize {
            if self.channel[c].channel_type != ChannelType::StereoPrimary
                || c + 1 >= self.config.channels as usize
            {
                continue;
            }

            for sf in 0..HCA_SUBFRAMES {
                let (energy_l, energy_r, energy_total) = self.stereo_energy(c, sf);
                let (quantized, energy_ratio) =
                    quantize_intensity(energy_l, energy_r, energy_total);
                self.channel[c + 1].intensity[sf] = quantized;
                self.mix_stereo_bands(c, sf, energy_ratio);
            }
        }
    }

    fn stereo_energy(&self, channel: usize, subframe: usize) -> (f32, f32, f32) {
        let mut left = 0.0;
        let mut right = 0.0;
        let mut total = 0.0;
        for band in self.base_band_count as usize..self.total_band_count as usize {
            let left_value = self.channel[channel].spectra[subframe][band];
            let right_value = self.channel[channel + 1].spectra[subframe][band];
            left += left_value.abs();
            right += right_value.abs();
            total += (left_value + right_value).abs();
        }
        (left, right, total * 2.0)
    }

    fn mix_stereo_bands(&mut self, channel: usize, subframe: usize, energy_ratio: f32) {
        for band in self.base_band_count as usize..self.total_band_count as usize {
            let mixed = (self.channel[channel].spectra[subframe][band]
                + self.channel[channel + 1].spectra[subframe][band])
                * energy_ratio;
            self.channel[channel].spectra[subframe][band] = mixed;
            self.channel[channel + 1].spectra[subframe][band] = 0.0;
        }
    }

    fn calculate_scale_factors(&mut self) {
        for c in 0..self.config.channels as usize {
            let coded_count = self.channel[c].coded_count;
            for b in 0..coded_count {
                let mut max_value = 0.0f32;
                for sf in 0..HCA_SUBFRAMES {
                    max_value = max_value.max(self.channel[c].spectra[sf][b].abs());
                }
                self.channel[c].scale_factors[b] = find_scale_factor(max_value);
            }
            self.channel[c].scale_factors[coded_count..HCA_SAMPLES_PER_SUBFRAME].fill(0);
        }
    }

    fn scale_spectra(&mut self) {
        for c in 0..self.config.channels as usize {
            let coded_count = self.channel[c].coded_count;
            for b in 0..coded_count {
                let scale_factor = self.channel[c].scale_factors[b] as usize;
                let scale = quantizer_scaling(scale_factor);
                for sf in 0..HCA_SUBFRAMES {
                    let value = self.channel[c].spectra[sf][b] * scale;
                    self.channel[c].scaled_spectra[b][sf] = if scale_factor == 0 {
                        0.0
                    } else {
                        value.clamp(-0.999_999_9, 0.999_999_9)
                    };
                }
            }
        }
    }

    fn calculate_hfr_group_averages(&mut self) {
        if self.hfr_group_count == 0 {
            return;
        }

        let hfr_start_band = (self.stereo_band_count + self.base_band_count) as usize;
        for c in 0..self.config.channels as usize {
            if self.channel[c].channel_type == ChannelType::StereoSecondary {
                continue;
            }
            self.calculate_channel_hfr_averages(c, hfr_start_band);
        }
    }

    fn calculate_channel_hfr_averages(&mut self, channel: usize, mut band: usize) {
        for group in 0..self.hfr_group_count as usize {
            let (average, next_band) = hfr_spectra_average(
                &self.channel[channel],
                band,
                self.bands_per_hfr_group as usize,
            );
            self.channel[channel].hfr_group_average_spectra[group] = average;
            band = next_band;
        }
    }

    fn calculate_hfr_scale(&mut self) {
        if self.hfr_group_count == 0 {
            return;
        }

        let hfr_start_band = (self.stereo_band_count + self.base_band_count) as usize;
        let hfr_band_count = self
            .hfr_band_count
            .min(self.total_band_count - self.hfr_band_count) as usize;

        for c in 0..self.config.channels as usize {
            if self.channel[c].channel_type == ChannelType::StereoSecondary {
                continue;
            }
            self.calculate_channel_hfr_scale(c, hfr_start_band, hfr_band_count);
        }
    }

    fn calculate_channel_hfr_scale(
        &mut self,
        channel: usize,
        hfr_start_band: usize,
        hfr_band_count: usize,
    ) {
        let mut band = 0;
        for group in 0..self.hfr_group_count as usize {
            let (average, next_band) = hfr_source_average(
                &self.channel[channel],
                hfr_start_band,
                hfr_band_count,
                band,
                self.bands_per_hfr_group as usize,
            );
            let mut group_spectra = self.channel[channel].hfr_group_average_spectra[group];
            if average > 0.0 {
                group_spectra *= (1.0 / average).min(SQRT_2);
            }
            self.channel[channel].hfr_scales[group] = find_scale_factor(group_spectra);
            band = next_band;
        }
    }

    fn calculate_frame_header_length(&mut self) {
        for c in 0..self.config.channels as usize {
            self.calculate_optimal_delta_length(c);
            if self.channel[c].channel_type == ChannelType::StereoSecondary {
                self.channel[c].header_length_bits += 4 * HCA_SUBFRAMES;
            } else if self.hfr_group_count > 0 {
                self.channel[c].header_length_bits += 6 * self.hfr_group_count as usize;
            }
        }
    }

    fn calculate_optimal_delta_length(&mut self, channel_index: usize) {
        let channel = &mut self.channel[channel_index];
        let coded_count = channel.coded_count;
        let empty_channel = channel.scale_factors[..coded_count]
            .iter()
            .all(|&scale| scale == 0);

        if empty_channel {
            channel.header_length_bits = 3;
            channel.scale_factor_delta_bits = 0;
            return;
        }

        let mut min_delta_bits = 6u8;
        let mut min_length = 3 + 6 * coded_count;

        for delta_bits in 1..6usize {
            let max_delta = (1 << (delta_bits - 1)) - 1;
            let mut length = 3 + 6;
            for band in 1..coded_count {
                let delta =
                    channel.scale_factors[band] as i32 - channel.scale_factors[band - 1] as i32;
                length += if delta.unsigned_abs() as usize > max_delta {
                    delta_bits + 6
                } else {
                    delta_bits
                };
            }

            if length < min_length {
                min_length = length;
                min_delta_bits = delta_bits as u8;
            }
        }

        channel.header_length_bits = min_length;
        channel.scale_factor_delta_bits = min_delta_bits;
    }

    fn calculate_noise_level(&mut self) -> Result<u32, HcaEncoderError> {
        let mut highest_band = (self.base_band_count + self.stereo_band_count) as i32 - 1;
        let available_bits = self.frame_size as usize * 8;
        let mut level = self.binary_search_level(available_bits, 0, 255);

        while level < 0 {
            highest_band -= 2;
            if highest_band < 0 {
                return Err(HcaEncoderError::FrameTooLarge);
            }

            for c in 0..self.config.channels as usize {
                for band in [highest_band + 1, highest_band + 2] {
                    if band >= 0 && (band as usize) < HCA_SAMPLES_PER_SUBFRAME {
                        self.channel[c].scale_factors[band as usize] = 0;
                    }
                }
            }

            self.calculate_frame_header_length();
            level = self.binary_search_level(available_bits, 0, 255);
        }

        Ok(level as u32)
    }

    fn binary_search_level(&self, available_bits: usize, mut low: i32, mut high: i32) -> i32 {
        let max = high;
        let mut mid_value = 0usize;

        while low != high {
            let mid = (low + high) / 2;
            mid_value = self.calculate_used_bits(mid, 0);

            if mid_value > available_bits {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        if low == max && mid_value > available_bits {
            -1
        } else {
            low
        }
    }

    fn calculate_evaluation_boundary(
        &self,
        acceptable_noise_level: u32,
    ) -> Result<u32, HcaEncoderError> {
        if acceptable_noise_level == 0 {
            return Ok(0);
        }

        let available_bits = self.frame_size as usize * 8;
        let level =
            self.binary_search_boundary(available_bits, acceptable_noise_level as i32, 0, 127);

        if level < 0 {
            Err(HcaEncoderError::FrameTooLarge)
        } else {
            Ok(level as u32)
        }
    }

    fn binary_search_boundary(
        &self,
        available_bits: usize,
        noise_level: i32,
        mut low: i32,
        mut high: i32,
    ) -> i32 {
        let max = high;

        while (high - low).abs() > 1 {
            let mid = (low + high) / 2;
            let mid_value = self.calculate_used_bits(noise_level, mid);
            if available_bits < mid_value {
                high = mid - 1;
            } else {
                low = mid;
            }
        }

        if low == high {
            return if low < max { low } else { -1 };
        }

        let high_value = self.calculate_used_bits(noise_level, high);
        if high_value > available_bits {
            low
        } else {
            high
        }
    }

    fn calculate_used_bits(&self, noise_level: i32, eval_boundary: i32) -> usize {
        let mut length = 16 + 16 + 16;

        for c in 0..self.config.channels as usize {
            let channel = &self.channel[c];
            length += channel.header_length_bits;

            for band in 0..channel.coded_count {
                let noise = if (band as i32) < eval_boundary {
                    noise_level - 1
                } else {
                    noise_level
                };
                let resolution = calculate_resolution(channel.scale_factors[band], noise) as usize;
                length += spectral_bit_count(channel, band, resolution);
            }
        }

        length
    }

    fn calculate_frame_resolutions(&mut self, noise_level: u32, evaluation_boundary: u32) {
        for c in 0..self.config.channels as usize {
            let coded_count = self.channel[c].coded_count;
            for band in 0..coded_count {
                let noise = if band < evaluation_boundary as usize {
                    noise_level.saturating_sub(1)
                } else {
                    noise_level
                };
                self.channel[c].resolutions[band] =
                    calculate_resolution(self.channel[c].scale_factors[band], noise as i32);
            }
            self.channel[c].resolutions[coded_count..HCA_SAMPLES_PER_SUBFRAME].fill(0);
        }
    }

    fn quantize_spectra(&mut self) {
        for c in 0..self.config.channels as usize {
            let coded_count = self.channel[c].coded_count;
            for band in 0..coded_count {
                let resolution = self.channel[c].resolutions[band] as usize;
                let step_size = QUANTIZER_INVERSE_STEP_SIZE[resolution];
                let shift_up = step_size + 1.0;
                let shift_down = (step_size + 0.5) as i32;

                for sf in 0..HCA_SUBFRAMES {
                    let quantized = (self.channel[c].scaled_spectra[band][sf] * step_size
                        + shift_up) as i32
                        - shift_down;
                    self.channel[c].quantized_spectra[sf][band] = if resolution < 8 {
                        quantized.clamp(-8, 7)
                    } else {
                        let bits = MAX_BIT_TABLE[resolution] as i32 - 1;
                        let max_value = (1 << bits) - 1;
                        quantized.clamp(-max_value, max_value)
                    };
                }
            }
        }
    }

    fn pack_frame(
        &self,
        acceptable_noise_level: u32,
        evaluation_boundary: u32,
    ) -> Result<Vec<u8>, HcaEncoderError> {
        let mut bits = BitWriter::new(self.frame_size as usize);
        bits.write(0xFFFF, 16);
        bits.write(acceptable_noise_level, 9);
        bits.write(evaluation_boundary, 7);

        for c in 0..self.config.channels as usize {
            self.write_scale_factors(&mut bits, c);
            if self.channel[c].channel_type == ChannelType::StereoSecondary {
                for sf in 0..HCA_SUBFRAMES {
                    bits.write(self.channel[c].intensity[sf] as u32, 4);
                }
            } else if self.hfr_group_count > 0 {
                for group in 0..self.hfr_group_count as usize {
                    bits.write(self.channel[c].hfr_scales[group] as u32, 6);
                }
            }
        }

        for sf in 0..HCA_SUBFRAMES {
            for c in 0..self.config.channels as usize {
                self.write_spectra(&mut bits, sf, c);
            }
        }

        if bits.position() > (self.frame_size as usize - 2) * 8 {
            return Err(HcaEncoderError::FrameTooLarge);
        }

        let mut frame = bits.into_vec();
        let checksum = crc16_checksum(&frame[..self.frame_size as usize - 2]);
        frame[self.frame_size as usize - 2] = (checksum >> 8) as u8;
        frame[self.frame_size as usize - 1] = (checksum & 0xFF) as u8;
        Ok(frame)
    }

    fn write_scale_factors(&self, writer: &mut BitWriter, channel_index: usize) {
        let channel = &self.channel[channel_index];
        let delta_bits = channel.scale_factor_delta_bits;
        writer.write(delta_bits as u32, 3);

        if delta_bits == 0 {
            return;
        }

        if delta_bits == 6 {
            for band in 0..channel.coded_count {
                writer.write(channel.scale_factors[band] as u32, 6);
            }
            return;
        }

        writer.write(channel.scale_factors[0] as u32, 6);
        let max_delta = (1 << (delta_bits - 1)) - 1;
        let escape_value = (1 << delta_bits) - 1;

        for band in 1..channel.coded_count {
            let delta = channel.scale_factors[band] as i32 - channel.scale_factors[band - 1] as i32;
            if delta.abs() > max_delta {
                writer.write(escape_value as u32, delta_bits as usize);
                writer.write(channel.scale_factors[band] as u32, 6);
            } else {
                writer.write((max_delta + delta) as u32, delta_bits as usize);
            }
        }
    }

    fn write_spectra(&self, writer: &mut BitWriter, subframe: usize, channel_index: usize) {
        let channel = &self.channel[channel_index];
        for band in 0..channel.coded_count {
            let resolution = channel.resolutions[band] as usize;
            if resolution == 0 {
                continue;
            }

            let quantized = channel.quantized_spectra[subframe][band];
            if resolution < 8 {
                let index = (quantized + 8).clamp(0, 15) as usize;
                let bits = QUANTIZE_SPECTRUM_BITS[resolution][index] as usize;
                let value = QUANTIZE_SPECTRUM_VALUE[resolution][index];
                writer.write(value, bits);
            } else {
                let bits = MAX_BIT_TABLE[resolution] as usize - 1;
                writer.write(quantized.unsigned_abs(), bits);
                if quantized != 0 {
                    writer.write(if quantized > 0 { 0 } else { 1 }, 1);
                }
            }
        }
    }
}

fn mdct_transform(channel: &mut ChannelEncodeState, subframe: usize) {
    let mut scratch = [0.0f32; HCA_SAMPLES_PER_SUBFRAME];
    let half = HCA_SAMPLES_PER_SUBFRAME / 2;

    for i in 0..half {
        let a = IMDCT_WINDOW[half - i - 1] * -channel.wave[subframe][half + i];
        let b = -IMDCT_WINDOW[half + i] * channel.wave[subframe][half - i - 1];
        let c = IMDCT_WINDOW[i] * channel.mdct_previous[i];
        let d = -IMDCT_WINDOW[HCA_SAMPLES_PER_SUBFRAME - i - 1]
            * channel.mdct_previous[HCA_SAMPLES_PER_SUBFRAME - i - 1];

        scratch[i] = a - b;
        scratch[half + i] = c - d;
    }

    for k in 0..HCA_SAMPLES_PER_SUBFRAME {
        let mut sum = 0.0f32;
        for (n, &sample) in scratch.iter().enumerate() {
            let angle = PI / HCA_SAMPLES_PER_SUBFRAME as f32 * (n as f32 + 0.5) * (k as f32 + 0.5);
            sum += sample * angle.cos();
        }
        channel.spectra[subframe][k] = sum * 0.125;
    }

    channel
        .mdct_previous
        .copy_from_slice(&channel.wave[subframe]);
}

fn find_scale_factor(value: f32) -> u8 {
    let mut low = 0usize;
    let mut high = 63usize;
    while low < high {
        let mid = (low + high) / 2;
        if DEQUANTIZER_SCALING_TABLE[mid] <= value {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low as u8
}

fn quantizer_scaling(scale_factor: usize) -> f32 {
    let dequant = DEQUANTIZER_SCALING_TABLE[scale_factor];
    if dequant > 0.0 {
        1.0 / dequant
    } else {
        0.0
    }
}

fn calculate_resolution(scale_factor: u8, noise_level: i32) -> u8 {
    if scale_factor == 0 {
        return 0;
    }

    let curve_position = (noise_level - 5 * scale_factor as i32 / 2 + 2).clamp(0, 58);
    SCALE_TO_RESOLUTION_CURVE[curve_position as usize].clamp(MIN_RESOLUTION, MAX_RESOLUTION)
}

fn ceil_div(a: u32, b: u32) -> u32 {
    if b == 0 {
        0
    } else {
        a.div_ceil(b)
    }
}

fn write_be_u16(data: &mut [u8], offset: usize, value: u16) {
    data[offset..offset + 2].copy_from_slice(&value.to_be_bytes());
}

fn write_be_u32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
}

fn mask_header(header: &mut [u8]) {
    xor_chunk_id(header, 0, 3);
    xor_chunk_id(header, 8, 3);
    xor_chunk_id(header, 24, 4);

    let mut position = 40usize;
    if unmasked_chunk(header, position) == *b"loop" {
        xor_chunk_id(header, position, 4);
        position += 16;
    }
    if unmasked_chunk(header, position) == *b"ciph" {
        xor_chunk_id(header, position, 4);
        position += 6;
    }
    if unmasked_chunk(header, position) == *b"pad\0" {
        xor_chunk_id(header, position, 3);
    }
}

fn unmasked_chunk(header: &[u8], offset: usize) -> [u8; 4] {
    [
        header[offset] & 0x7F,
        header[offset + 1] & 0x7F,
        header[offset + 2] & 0x7F,
        header[offset + 3] & 0x7F,
    ]
}

fn xor_chunk_id(header: &mut [u8], offset: usize, count: usize) {
    for i in 0..count {
        header[offset + i] ^= 0x80;
    }
}

/// Convenience function to encode WAV to HCA.
pub fn encode_wav_to_hca<W: Write + Seek>(
    wav_data: &[u8],
    writer: &mut W,
    config: Option<HcaEncoderConfig>,
) -> Result<(), HcaEncoderError> {
    if wav_data.len() < 44 || &wav_data[0..4] != b"RIFF" || &wav_data[8..12] != b"WAVE" {
        return Err(HcaEncoderError::NoSamples);
    }

    let mut pos = 12;
    let mut channels = 2u32;
    let mut sample_rate = 44100u32;
    let mut bits_per_sample = 16u32;
    let mut data_start = 0;
    let mut data_len = 0;

    while pos + 8 <= wav_data.len() {
        let chunk_id = &wav_data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            wav_data[pos + 4],
            wav_data[pos + 5],
            wav_data[pos + 6],
            wav_data[pos + 7],
        ]) as usize;

        if chunk_id == b"fmt " && chunk_size >= 16 {
            channels = u16::from_le_bytes([wav_data[pos + 10], wav_data[pos + 11]]) as u32;
            sample_rate = u32::from_le_bytes([
                wav_data[pos + 12],
                wav_data[pos + 13],
                wav_data[pos + 14],
                wav_data[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([wav_data[pos + 22], wav_data[pos + 23]]) as u32;
        } else if chunk_id == b"data" {
            data_start = pos + 8;
            data_len = chunk_size;
            break;
        }

        pos += 8 + chunk_size + (chunk_size & 1);
    }

    if data_start == 0 || data_len == 0 {
        return Err(HcaEncoderError::NoSamples);
    }

    let mut samples = Vec::new();
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let sample_count = data_len / bytes_per_sample;

    for i in 0..sample_count {
        let offset = data_start + i * bytes_per_sample;
        if offset + bytes_per_sample > wav_data.len() {
            break;
        }

        let sample = match bits_per_sample {
            16 => {
                let raw = i16::from_le_bytes([wav_data[offset], wav_data[offset + 1]]);
                raw as f32 / 32768.0
            }
            24 => {
                let raw = ((wav_data[offset] as i32)
                    | ((wav_data[offset + 1] as i32) << 8)
                    | ((wav_data[offset + 2] as i32) << 16))
                    << 8
                    >> 8;
                raw as f32 / 8_388_608.0
            }
            32 => f32::from_le_bytes([
                wav_data[offset],
                wav_data[offset + 1],
                wav_data[offset + 2],
                wav_data[offset + 3],
            ]),
            _ => 0.0,
        };
        samples.push(sample);
    }

    let cfg = config.unwrap_or_else(|| HcaEncoderConfig::new(sample_rate, channels));
    let mut encoder = HcaEncoder::new(cfg)?;
    encoder.encode(&samples, writer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config() {
        let config = HcaEncoderConfig::new(44100, 2).with_bitrate(256000);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bitrate, 256000);
    }

    #[test]
    fn test_encoder_creation() {
        let config = HcaEncoderConfig::new(44100, 2);
        let encoder = HcaEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_intensity_and_hfr_helpers() {
        assert_eq!(quantize_intensity(0.0, 0.0, 0.0), (0, 1.0));
        let (intensity, ratio) = quantize_intensity(3.0, 1.0, 2.0);
        assert!((1..=13).contains(&intensity));
        assert!((0.5..=SQRT_2 / 2.0).contains(&ratio));

        let mut channel = ChannelEncodeState::default();
        for subframe in 0..HCA_SUBFRAMES {
            channel.spectra[subframe][126] = 2.0;
            channel.spectra[subframe][127] = 4.0;
            channel.scaled_spectra[3][subframe] = 2.0;
            channel.scaled_spectra[2][subframe] = 4.0;
        }
        assert_eq!(hfr_spectra_average(&channel, 126, 4), (3.0, 128));
        assert_eq!(hfr_spectra_average(&channel, 128, 1), (0.0, 128));
        assert_eq!(hfr_source_average(&channel, 4, 4, 0, 2), (3.0, 2));
        assert_eq!(hfr_source_average(&channel, 4, 0, 0, 1), (0.0, 0));

        let mut encoder = HcaEncoder::new(HcaEncoderConfig::new(44100, 2)).unwrap();
        encoder.hfr_group_count = 2;
        encoder.bands_per_hfr_group = 1;
        encoder.channel[0] = channel;
        encoder.calculate_channel_hfr_averages(0, 126);
        assert_eq!(
            encoder.channel[0].hfr_group_average_spectra[..2],
            [2.0, 4.0]
        );
        encoder.calculate_channel_hfr_scale(0, 4, 4);
        assert!(encoder.channel[0].hfr_scales[0] > 0);
        assert!(encoder.channel[0].hfr_scales[1] > 0);
    }

    #[test]
    fn test_invalid_config() {
        let config = HcaEncoderConfig::new(0, 2);
        let encoder = HcaEncoder::new(config);
        assert!(encoder.is_err());

        let config = HcaEncoderConfig::new(44100, 0);
        let encoder = HcaEncoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_crc16() {
        let data = [0x48, 0x43, 0x41, 0x00];
        let crc = crc16_checksum(&data);
        assert!(crc != 0);
    }

    #[test]
    fn test_encode_simple() {
        let config = HcaEncoderConfig::new(44100, 1);
        let mut encoder = HcaEncoder::new(config).unwrap();

        let samples: Vec<f32> = (0..HCA_SAMPLES_PER_FRAME)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();

        let mut output = std::io::Cursor::new(Vec::new());
        let result = encoder.encode(&samples, &mut output);
        assert!(result.is_ok());

        let data = output.into_inner();
        assert!(!data.is_empty());
        assert!(data[0] == 0x48 || data[0] == 0xC8);
    }

    #[test]
    fn test_encode_encrypted_roundtrip() {
        let key = 0x1234_5678_90AB_CDEF;
        let config = HcaEncoderConfig::new(44100, 1).with_encryption(key);
        let mut encoder = HcaEncoder::new(config).unwrap();

        let samples: Vec<f32> = (0..HCA_SAMPLES_PER_FRAME * 2)
            .map(|i| (2.0 * PI * 330.0 * i as f32 / 44100.0).sin() * 0.4)
            .collect();

        let mut output = std::io::Cursor::new(Vec::new());
        encoder.encode(&samples, &mut output).unwrap();
        let data = output.into_inner();
        assert_eq!(data[0], 0xC8);

        let mut decoder = crate::hca::HcaDecoder::from_reader(std::io::Cursor::new(data)).unwrap();
        decoder.set_encryption_key(key, 0);
        let decoded = decoder.decode_all().unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_decode_with_keycode_and_subkey() {
        // pjsk-style type-56 HCA is keyed by a global keycode + a per-AWB subkey;
        // HcaDecoder::set_encryption_key combines them into the effective key. Encode
        // with that effective key, then verify decode via keycode+subkey succeeds.
        // enc/dec are nested fns so each large ClHca struct lives in its own stack
        // frame, reclaimed on return (debug builds don't reuse block-local slots,
        // so several decoders in one frame would blow the 2 MB test-thread stack).
        fn enc(samples: &[f32], key: Option<u64>) -> Vec<u8> {
            let mut cfg = HcaEncoderConfig::new(44100, 1);
            if let Some(k) = key {
                cfg = cfg.with_encryption(k);
            }
            let mut e = HcaEncoder::new(cfg).unwrap();
            let mut o = std::io::Cursor::new(Vec::new());
            e.encode(samples, &mut o).unwrap();
            o.into_inner()
        }
        fn dec(data: Vec<u8>, key: Option<(u64, u64)>) -> Vec<f32> {
            let mut d = crate::hca::HcaDecoder::from_reader(std::io::Cursor::new(data)).unwrap();
            if let Some((k, s)) = key {
                d.set_encryption_key(k, s);
            }
            d.decode_all().unwrap()
        }

        let keycode: u64 = 0x0011_2233_4455_6677;
        let subkey: u16 = 0x1234;
        let multiplier = ((subkey as u64) << 16) | ((!subkey as u64).wrapping_add(2));
        let effective = keycode.wrapping_mul(multiplier);

        let samples: Vec<f32> = (0..HCA_SAMPLES_PER_FRAME * 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();

        let encrypted = enc(&samples, Some(effective));
        assert_eq!(encrypted[0], 0xC8); // encrypted HCA magic (high bit set)

        // Decode the encrypted stream via keycode+subkey; must match the plain encode.
        let decrypted = dec(encrypted, Some((keycode, subkey as u64)));
        let plain = dec(enc(&samples, None), None);

        assert!(!decrypted.is_empty());
        assert_eq!(plain.len(), decrypted.len());
        let max_diff = plain
            .iter()
            .zip(&decrypted)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff == 0.0, "keycode+subkey must recover identical PCM");
    }
}
