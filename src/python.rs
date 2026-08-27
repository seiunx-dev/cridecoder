//! Python bindings for cridecoder
//!
//! Provides Python functions for CRI codec operations:
//! - ACB extraction and building
//! - HCA decoding and encoding
//! - USM extraction and building

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use std::fs;
use std::io::Cursor;
use std::path::Path;

use crate::acb;
use crate::acb::{AcbBuilder, TrackInput};
use crate::hca::{HcaDecoder, HcaEncoder, HcaEncoderConfig};
use crate::usm;
use crate::usm::UsmBuilder;

/// Extract audio tracks from an ACB file.
///
/// Args:
///     acb_path: Path to the ACB file
///     output_dir: Directory to write extracted files to
///
/// Returns:
///     List of extracted file paths, or None if the file is invalid
#[pyfunction]
fn extract_acb(py: Python<'_>, acb_path: &str, output_dir: &str) -> PyResult<Option<Vec<String>>> {
    let acb_path = Path::new(acb_path);
    let output_dir = Path::new(output_dir);
    fs::create_dir_all(output_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output dir: {}", e)))?;

    py.detach(|| acb::extract_acb_from_file(acb_path, output_dir))
        .map_err(|e| PyRuntimeError::new_err(format!("ACB extraction failed: {}", e)))
}

/// Extract audio tracks from an ACB file, returning per-track metadata.
///
/// Unlike :func:`extract_acb`, this also surfaces each track's cue id and the
/// AFS2 subkey of the AWB it came from, which is required (together with the
/// global keycode) to decode type-56 encrypted HCA.
///
/// Args:
///     acb_path: Path to the ACB file
///     output_dir: Directory to write extracted files to
///
/// Returns:
///     List of dicts ``{"path", "name", "cue_id", "subkey"}``, or None if the
///     file is invalid.
#[pyfunction]
fn extract_acb_tracks<'py>(
    py: Python<'py>,
    acb_path: &str,
    output_dir: &str,
) -> PyResult<Option<Vec<Bound<'py, pyo3::types::PyDict>>>> {
    let acb_path = Path::new(acb_path);
    let output_dir = Path::new(output_dir);
    fs::create_dir_all(output_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output dir: {}", e)))?;

    let tracks = py
        .detach(|| acb::extract_acb_tracks_from_file(acb_path, output_dir))
        .map_err(|e| PyRuntimeError::new_err(format!("ACB extraction failed: {}", e)))?;

    let tracks = match tracks {
        Some(t) => t,
        None => return Ok(None),
    };

    let mut out = Vec::with_capacity(tracks.len());
    for track in tracks {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("path", track.path)?;
        dict.set_item("name", track.name)?;
        dict.set_item("cue_id", track.cue_id)?;
        dict.set_item("subkey", track.subkey)?;
        out.push(dict);
    }
    Ok(Some(out))
}

/// Extract an ACB and decode its HCA tracks straight to WAV files.
///
/// The per-AWB AFS2 subkey is read and applied automatically, so encrypted
/// (type-56) ACBs only need the global ``key``. Non-HCA tracks are written out
/// verbatim with their original extension.
///
/// Args:
///     acb_path: Path to the ACB file
///     output_dir: Directory to write the decoded WAV (and any raw) files to
///     key: Global HCA keycode (omit/None for unencrypted ACBs)
///
/// Returns:
///     List of written file paths.
#[pyfunction]
#[pyo3(signature = (acb_path, output_dir, key=None, threads=None))]
fn decode_acb_to_wav(
    py: Python<'_>,
    acb_path: &str,
    output_dir: &str,
    key: Option<u64>,
    threads: Option<usize>,
) -> PyResult<Vec<String>> {
    let out_dir = Path::new(output_dir);
    fs::create_dir_all(out_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output dir: {}", e)))?;

    let threads = resolve_threads(threads);
    py.detach(|| {
        acb::decode_acb_to_wav_from_file_parallel(Path::new(acb_path), out_dir, key, threads)
    })
    .map_err(|e| PyRuntimeError::new_err(format!("ACB decode failed: {}", e)))
}

/// Extract audio tracks from in-memory ACB bytes (no disk I/O).
///
/// The in-memory counterpart of :func:`extract_acb_tracks`: takes the ACB
/// bytes directly and returns the waveform bytes per track instead of writing
/// files. Only the embedded AWB is read — external streaming ``.awb`` archives
/// can't be resolved without a path, so use :func:`extract_acb` for those.
///
/// Args:
///     acb_data: Raw ACB file bytes
///
/// Returns:
///     List of dicts ``{"name", "cue_id", "extension", "subkey", "data"}``.
#[pyfunction]
fn extract_acb_bytes<'py>(
    py: Python<'py>,
    acb_data: &[u8],
) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let tracks = py
        .detach(|| acb::extract_acb_to_memory(Cursor::new(acb_data), None))
        .map_err(|e| PyRuntimeError::new_err(format!("ACB extraction failed: {}", e)))?;

    let mut out = Vec::with_capacity(tracks.len());
    for track in tracks {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("name", track.name)?;
        dict.set_item("cue_id", track.cue_id)?;
        dict.set_item("extension", track.extension)?;
        dict.set_item("subkey", track.subkey)?;
        dict.set_item("data", pyo3::types::PyBytes::new(py, &track.data))?;
        out.push(dict);
    }
    Ok(out)
}

/// Extract each distinct waveform from in-memory ACB bytes exactly once.
///
/// ACBs often point several cues at the same physical waveform; unlike
/// :func:`extract_acb_bytes` (which copies it once per cue), this reads and
/// copies each waveform a single time and lists the cues that reference it.
///
/// Args:
///     acb_data: Raw ACB file bytes
///
/// Returns:
///     List of dicts ``{"extension", "subkey", "data", "cues"}`` where ``cues``
///     is a list of ``{"name", "cue_id"}`` (at least one).
#[pyfunction]
fn extract_acb_unique_bytes<'py>(
    py: Python<'py>,
    acb_data: &[u8],
) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let waveforms = py
        .detach(|| acb::extract_acb_unique_to_memory(Cursor::new(acb_data), None))
        .map_err(|e| PyRuntimeError::new_err(format!("ACB extraction failed: {}", e)))?;

    let mut out = Vec::with_capacity(waveforms.len());
    for wf in waveforms {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("extension", wf.extension)?;
        dict.set_item("subkey", wf.subkey)?;
        dict.set_item("data", pyo3::types::PyBytes::new(py, &wf.data))?;
        let mut cues = Vec::with_capacity(wf.cues.len());
        for cue in wf.cues {
            let c = pyo3::types::PyDict::new(py);
            c.set_item("name", cue.name)?;
            c.set_item("cue_id", cue.cue_id)?;
            cues.push(c);
        }
        dict.set_item("cues", cues)?;
        out.push(dict);
    }
    Ok(out)
}

/// Decode an in-memory ACB straight to WAV bytes (no disk I/O).
///
/// The in-memory counterpart of :func:`decode_acb_to_wav`: each AWB's subkey is
/// applied automatically, so encrypted ACBs only need the global ``key``.
/// Non-HCA tracks are returned verbatim (``extension`` reflects this).
///
/// Args:
///     acb_data: Raw ACB file bytes
///     key: Global HCA keycode (omit/None for unencrypted ACBs)
///
/// Returns:
///     List of dicts ``{"name", "cue_id", "extension", "data"}`` where ``data``
///     is WAV bytes for HCA tracks (``extension == "wav"``).
#[pyfunction]
#[pyo3(signature = (acb_data, key=None, threads=None))]
fn decode_acb_to_wav_bytes<'py>(
    py: Python<'py>,
    acb_data: &[u8],
    key: Option<u64>,
    threads: Option<usize>,
) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let threads = resolve_threads(threads);
    let tracks = py
        .detach(|| {
            acb::decode_acb_to_wav_to_memory_parallel(Cursor::new(acb_data), None, key, threads)
        })
        .map_err(|e| PyRuntimeError::new_err(format!("ACB decode failed: {}", e)))?;

    let mut out = Vec::with_capacity(tracks.len());
    for track in tracks {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("name", track.name)?;
        dict.set_item("cue_id", track.cue_id)?;
        dict.set_item("extension", track.extension)?;
        dict.set_item("data", pyo3::types::PyBytes::new(py, &track.data))?;
        out.push(dict);
    }
    Ok(out)
}

/// Build an ACB file from track data.
///
/// Args:
///     tracks: List of tuples (name, cue_id, hca_data)
///     output_path: Path to write the ACB file
///
/// Returns:
///     None on success
#[pyfunction]
fn build_acb(
    py: Python<'_>,
    tracks: Vec<(String, u32, Vec<u8>)>,
    output_path: &str,
) -> PyResult<()> {
    let mut builder = AcbBuilder::new();

    for (name, cue_id, data) in tracks {
        let track = TrackInput::new(name, cue_id, data);
        builder.add_track(track);
    }

    let mut output = fs::File::create(output_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output file: {}", e)))?;

    py.detach(|| builder.build(&mut output, None))
        .map_err(|e| PyRuntimeError::new_err(format!("ACB build failed: {}", e)))?;

    Ok(())
}

/// Build an ACB file from track data (returns bytes).
///
/// Args:
///     tracks: List of tuples (name, cue_id, hca_data)
///
/// Returns:
///     ACB file data as bytes
#[pyfunction]
fn build_acb_bytes(py: Python<'_>, tracks: Vec<(String, u32, Vec<u8>)>) -> PyResult<Vec<u8>> {
    let mut builder = AcbBuilder::new();

    for (name, cue_id, data) in tracks {
        let track = TrackInput::new(name, cue_id, data);
        builder.add_track(track);
    }

    py.detach(|| {
        let mut output = Cursor::new(Vec::new());
        builder
            .build(&mut output, None)
            .map(|_| output.into_inner())
    })
    .map_err(|e| PyRuntimeError::new_err(format!("ACB build failed: {}", e)))
}

/// Build a single-track music ACB from one HCA track (returns bytes).
///
/// Args:
///     name: Cue sheet/base cue name
///     hca_data: Raw HCA file data
///     cue_id: Base cue ID
///     virtual_cue_suffix: Optional suffix for the paired virtual cue
///     memory_awb_id: Embedded AWB file ID
///     reference_num_samples: Fallback/reference sample count
///     reference_length_ms: Fallback/reference cue length in milliseconds
///     acb_version: Header version value
///     acf_md5_hash: 16-byte ACF hash
///     acb_guid: 16-byte ACB GUID
///     version_string: Header version string
///     acb_volume: Header volume value
///     category_extension: Category extension value
///     cue_priority_type: Cue priority type value
///     acf_category_name: ACF category reference name
///     acf_category_id: ACF category reference id
///     acf_bus_names: ACF bus/output names
///
/// Returns:
///     ACB file data as bytes
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn build_music_acb_bytes(
    py: Python<'_>,
    name: String,
    hca_data: Vec<u8>,
    cue_id: u32,
    virtual_cue_suffix: Option<String>,
    memory_awb_id: u16,
    reference_num_samples: u32,
    reference_length_ms: u32,
    acb_version: u32,
    acf_md5_hash: Vec<u8>,
    acb_guid: Vec<u8>,
    version_string: String,
    acb_volume: f32,
    category_extension: u8,
    cue_priority_type: u8,
    acf_category_name: String,
    acf_category_id: u32,
    acf_bus_names: Vec<String>,
) -> PyResult<Vec<u8>> {
    let mut builder = AcbBuilder::new().music_acb(
        cue_id,
        virtual_cue_suffix,
        memory_awb_id,
        reference_num_samples,
        reference_length_ms,
        acb_version,
        acf_md5_hash,
        acb_guid,
        version_string,
        acb_volume,
        category_extension,
        cue_priority_type,
        acf_category_name,
        acf_category_id,
        acf_bus_names,
    );
    builder.add_track(TrackInput::new(name, cue_id, hca_data));

    py.detach(|| {
        let mut output = Cursor::new(Vec::new());
        builder
            .build(&mut output, None)
            .map(|_| output.into_inner())
    })
    .map_err(|e| PyRuntimeError::new_err(format!("Music ACB build failed: {}", e)))
}

/// Resolve the optional `threads` argument shared by the HCA decode bindings:
/// `None` -> 1 (serial), `0` -> all available cores, `n` -> n.
fn resolve_threads(threads: Option<usize>) -> usize {
    match threads {
        None => 1,
        Some(0) => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        Some(n) => n,
    }
}

/// Decode an HCA file to WAV format.
///
/// Args:
///     hca_path: Path to the HCA file
///     wav_path: Path to write the output WAV file
///     threads: None = serial decode; 0 = use all CPU cores; N = use N threads.
///         Multithreaded output is byte-identical to serial.
///
/// Returns:
///     dict with HCA info (sample_rate, channels, block_count, etc.)
#[pyfunction]
#[pyo3(signature = (hca_path, wav_path, key=None, subkey=None, threads=None))]
fn decode_hca<'py>(
    py: Python<'py>,
    hca_path: &str,
    wav_path: &str,
    key: Option<u64>,
    subkey: Option<u64>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let mut decoder = HcaDecoder::from_file(hca_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open HCA: {}", e)))?;
    // Apply the decryption key for type-56 encrypted HCA (no-op for unencrypted files).
    if let Some(k) = key {
        decoder.set_encryption_key(k, subkey.unwrap_or(0));
    }

    let info = decoder.info().clone();

    let mut output = fs::File::create(wav_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create WAV: {}", e)))?;

    let threads = resolve_threads(threads);
    py.detach(|| decoder.decode_to_wav_parallel(&mut output, threads))
        .map_err(|e| PyRuntimeError::new_err(format!("HCA decode failed: {}", e)))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("sample_rate", info.sampling_rate)?;
    dict.set_item("channels", info.channel_count)?;
    dict.set_item("block_count", info.block_count)?;
    dict.set_item("block_size", info.block_size)?;
    dict.set_item("encoder_delay", info.encoder_delay)?;
    dict.set_item("samples_per_block", info.samples_per_block)?;

    Ok(dict)
}

/// Decode HCA data (bytes) to WAV bytes in memory.
///
/// Args:
///     hca_data: Raw HCA file data as bytes
///     threads: None = serial decode; 0 = use all CPU cores; N = use N threads.
///         Multithreaded output is byte-identical to serial.
///
/// Returns:
///     WAV file data as bytes
#[pyfunction]
#[pyo3(signature = (hca_data, key=None, subkey=None, threads=None))]
fn decode_hca_bytes(
    py: Python<'_>,
    hca_data: &[u8],
    key: Option<u64>,
    subkey: Option<u64>,
    threads: Option<usize>,
) -> PyResult<Vec<u8>> {
    let mut decoder = HcaDecoder::from_reader(Cursor::new(hca_data))
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse HCA: {}", e)))?;
    if let Some(k) = key {
        decoder.set_encryption_key(k, subkey.unwrap_or(0));
    }

    let threads = resolve_threads(threads);
    let mut wav_buf = Vec::new();
    py.detach(|| decoder.decode_to_wav_parallel(&mut wav_buf, threads))
        .map_err(|e| PyRuntimeError::new_err(format!("HCA decode failed: {}", e)))?;

    Ok(wav_buf)
}

/// Encode WAV data to HCA format.
///
/// Args:
///     wav_data: WAV file data as bytes
///     sample_rate: Sample rate (optional, auto-detect from WAV if None)
///     channels: Number of channels (optional, auto-detect from WAV if None)
///     bitrate: Target bitrate in bps (default: 256000)
///     encryption_key: Optional encryption key (u64)
///
/// Returns:
///     HCA file data as bytes
#[pyfunction]
#[pyo3(signature = (wav_data, sample_rate=None, channels=None, bitrate=256000, encryption_key=None))]
fn encode_hca_bytes(
    py: Python<'_>,
    wav_data: &[u8],
    sample_rate: Option<u32>,
    channels: Option<u32>,
    bitrate: u32,
    encryption_key: Option<u64>,
) -> PyResult<Vec<u8>> {
    py.detach(|| encode_wav_to_hca(wav_data, sample_rate, channels, bitrate, encryption_key))
}

/// GIL-free core of the HCA encode bindings (WAV parse + PCM convert + encode).
fn encode_wav_to_hca(
    wav_data: &[u8],
    sample_rate: Option<u32>,
    channels: Option<u32>,
    bitrate: u32,
    encryption_key: Option<u64>,
) -> PyResult<Vec<u8>> {
    // Parse WAV header
    if wav_data.len() < 44 || &wav_data[0..4] != b"RIFF" || &wav_data[8..12] != b"WAVE" {
        return Err(PyRuntimeError::new_err("Invalid WAV data"));
    }

    // Find fmt and data chunks
    let mut pos = 12;
    let mut wav_channels = 2u32;
    let mut wav_sample_rate = 44100u32;
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
            wav_channels = u16::from_le_bytes([wav_data[pos + 10], wav_data[pos + 11]]) as u32;
            wav_sample_rate = u32::from_le_bytes([
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

        pos += 8 + chunk_size;
        if !chunk_size.is_multiple_of(2) {
            pos += 1; // padding
        }
    }

    if data_start == 0 || data_len == 0 {
        return Err(PyRuntimeError::new_err("No data chunk in WAV"));
    }

    let final_sample_rate = sample_rate.unwrap_or(wav_sample_rate);
    let final_channels = channels.unwrap_or(wav_channels);

    // Convert PCM to f32
    let samples: Vec<f32> = match bits_per_sample {
        16 => {
            let sample_count = data_len / 2;
            (0..sample_count)
                .map(|i| {
                    let idx = data_start + i * 2;
                    let sample = i16::from_le_bytes([wav_data[idx], wav_data[idx + 1]]);
                    sample as f32 / 32768.0
                })
                .collect()
        }
        24 => {
            let sample_count = data_len / 3;
            (0..sample_count)
                .map(|i| {
                    let idx = data_start + i * 3;
                    let sample = ((wav_data[idx] as i32)
                        | ((wav_data[idx + 1] as i32) << 8)
                        | ((wav_data[idx + 2] as i32) << 16))
                        << 8
                        >> 8; // sign extend
                    sample as f32 / 8388608.0
                })
                .collect()
        }
        32 => {
            let sample_count = data_len / 4;
            (0..sample_count)
                .map(|i| {
                    let idx = data_start + i * 4;
                    f32::from_le_bytes([
                        wav_data[idx],
                        wav_data[idx + 1],
                        wav_data[idx + 2],
                        wav_data[idx + 3],
                    ])
                })
                .collect()
        }
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported bit depth: {}",
                bits_per_sample
            )))
        }
    };

    // Create encoder config
    let mut config = HcaEncoderConfig::new(final_sample_rate, final_channels).with_bitrate(bitrate);

    if let Some(key) = encryption_key {
        config = config.with_encryption(key);
    }

    // Encode
    let mut encoder = HcaEncoder::new(config)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create encoder: {}", e)))?;

    let mut output = Cursor::new(Vec::new());
    encoder
        .encode(&samples, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("HCA encode failed: {}", e)))?;

    Ok(output.into_inner())
}

/// Encode a WAV file to HCA file.
///
/// Args:
///     wav_path: Path to the input WAV file
///     hca_path: Path to write the output HCA file
///     bitrate: Target bitrate in bps (default: 256000)
///     encryption_key: Optional encryption key (u64)
///
/// Returns:
///     dict with encoding info
#[pyfunction]
#[pyo3(signature = (wav_path, hca_path, bitrate=256000, encryption_key=None))]
fn encode_hca<'py>(
    py: Python<'py>,
    wav_path: &str,
    hca_path: &str,
    bitrate: u32,
    encryption_key: Option<u64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let hca_data = py.detach(|| {
        let wav_data = fs::read(wav_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read WAV: {}", e)))?;

        let hca_data = encode_wav_to_hca(&wav_data, None, None, bitrate, encryption_key)?;

        fs::write(hca_path, &hca_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write HCA: {}", e)))?;
        Ok::<_, PyErr>(hca_data)
    })?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("size", hca_data.len())?;
    dict.set_item("bitrate", bitrate)?;

    Ok(dict)
}

/// Extract video/audio from a USM file.
///
/// Args:
///     usm_path: Path to the USM file
///     output_dir: Directory to write extracted files to
///     key: Optional decryption key (u64)
///     export_audio: Whether to export audio tracks (default: false)
///
/// Returns:
///     List of extracted file paths
#[pyfunction]
#[pyo3(signature = (usm_path, output_dir, key=None, export_audio=false))]
fn extract_usm(
    py: Python<'_>,
    usm_path: &str,
    output_dir: &str,
    key: Option<u64>,
    export_audio: bool,
) -> PyResult<Vec<String>> {
    let usm_path = Path::new(usm_path);
    let output_dir = Path::new(output_dir);
    fs::create_dir_all(output_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output dir: {}", e)))?;

    let files = py
        .detach(|| usm::extract_usm_file(usm_path, output_dir, key, export_audio))
        .map_err(|e| PyRuntimeError::new_err(format!("USM extraction failed: {}", e)))?;

    Ok(files
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect())
}

/// Extract USM streams from in-memory bytes (no disk I/O).
///
/// The in-memory counterpart of :func:`extract_usm`: takes the USM bytes and
/// returns each stream's bytes instead of writing files.
///
/// Args:
///     usm_data: Raw USM file bytes
///     key: Optional decryption key (u64)
///     export_audio: Whether to include audio streams (default: false)
///
/// Returns:
///     List of dicts ``{"name", "extension", "data"}`` (video, and audio when
///     ``export_audio`` is set).
#[pyfunction]
#[pyo3(signature = (usm_data, key=None, export_audio=false))]
fn extract_usm_bytes<'py>(
    py: Python<'py>,
    usm_data: &[u8],
    key: Option<u64>,
    export_audio: bool,
) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let streams = py
        .detach(|| usm::extract_usm_to_memory(Cursor::new(usm_data), b"", key, export_audio))
        .map_err(|e| PyRuntimeError::new_err(format!("USM extraction failed: {}", e)))?;

    let mut out = Vec::with_capacity(streams.len());
    for stream in streams {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("name", stream.name)?;
        dict.set_item("extension", stream.extension)?;
        dict.set_item("data", pyo3::types::PyBytes::new(py, &stream.data))?;
        out.push(dict);
    }
    Ok(out)
}

/// Build a USM file from video data.
///
/// Args:
///     name: Name for the USM file (used in metadata)
///     video_data: M2V video data as bytes
///     output_path: Path to write the USM file
///     encryption_key: Optional encryption key (u64)
///
/// Returns:
///     None on success
#[pyfunction]
#[pyo3(signature = (name, video_data, output_path, encryption_key=None))]
fn build_usm(
    py: Python<'_>,
    name: &str,
    video_data: Vec<u8>,
    output_path: &str,
    encryption_key: Option<u64>,
) -> PyResult<()> {
    let mut builder = UsmBuilder::new(name.to_string()).video(video_data);

    if let Some(key) = encryption_key {
        builder = builder.encryption_key(key);
    }

    let mut output = fs::File::create(output_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output file: {}", e)))?;

    py.detach(|| builder.build(&mut output))
        .map_err(|e| PyRuntimeError::new_err(format!("USM build failed: {}", e)))?;

    Ok(())
}

/// Build a USM file from video data (returns bytes).
///
/// Args:
///     name: Name for the USM file (used in metadata)
///     video_data: M2V video data as bytes
///     encryption_key: Optional encryption key (u64)
///
/// Returns:
///     USM file data as bytes
#[pyfunction]
#[pyo3(signature = (name, video_data, encryption_key=None))]
fn build_usm_bytes(
    py: Python<'_>,
    name: &str,
    video_data: Vec<u8>,
    encryption_key: Option<u64>,
) -> PyResult<Vec<u8>> {
    let mut builder = UsmBuilder::new(name.to_string()).video(video_data);

    if let Some(key) = encryption_key {
        builder = builder.encryption_key(key);
    }

    py.detach(|| {
        let mut output = Cursor::new(Vec::new());
        builder.build(&mut output).map(|_| output.into_inner())
    })
    .map_err(|e| PyRuntimeError::new_err(format!("USM build failed: {}", e)))
}

/// Read metadata from a USM file.
///
/// Args:
///     usm_path: Path to the USM file
///
/// Returns:
///     Metadata as a JSON string
#[pyfunction]
fn read_usm_metadata(py: Python<'_>, usm_path: &str) -> PyResult<String> {
    let usm_path = Path::new(usm_path);
    let metadata = py
        .detach(|| usm::read_metadata_file(usm_path))
        .map_err(|e| PyRuntimeError::new_err(format!("Metadata read failed: {}", e)))?;

    serde_json::to_string_pretty(&metadata)
        .map_err(|e| PyRuntimeError::new_err(format!("JSON serialization failed: {}", e)))
}

/// Register all Python functions to the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ACB functions
    m.add_function(wrap_pyfunction!(extract_acb, m)?)?;
    m.add_function(wrap_pyfunction!(extract_acb_tracks, m)?)?;
    m.add_function(wrap_pyfunction!(extract_acb_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(extract_acb_unique_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(decode_acb_to_wav, m)?)?;
    m.add_function(wrap_pyfunction!(decode_acb_to_wav_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(build_acb, m)?)?;
    m.add_function(wrap_pyfunction!(build_acb_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(build_music_acb_bytes, m)?)?;

    // HCA functions
    m.add_function(wrap_pyfunction!(decode_hca, m)?)?;
    m.add_function(wrap_pyfunction!(decode_hca_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(encode_hca, m)?)?;
    m.add_function(wrap_pyfunction!(encode_hca_bytes, m)?)?;

    // USM functions
    m.add_function(wrap_pyfunction!(extract_usm, m)?)?;
    m.add_function(wrap_pyfunction!(extract_usm_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(build_usm, m)?)?;
    m.add_function(wrap_pyfunction!(build_usm_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(read_usm_metadata, m)?)?;

    Ok(())
}
