//! ACB file extractor

use crate::acb::afs::AfsArchive;
use crate::acb::consts::wave_type_extension;
use crate::acb::track::{Track, TrackList};
use crate::acb::utf::{get_bytes_field, get_string_field, take_bytes_field, UtfTable};
use std::collections::HashMap;
use std::fs;
use std::io::{Cursor, Read, Seek};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedAcbTrack {
    pub name: String,
    /// Cue id of the track (cue-table index).
    pub cue_id: i32,
    pub extension: String,
    pub data: Vec<u8>,
    /// AFS2 subkey of the AWB this waveform came from. Required (together with the
    /// global keycode) to decrypt type-56 encrypted HCA via
    /// `HcaDecoder::set_encryption_key`. 0 when the AWB is unencrypted.
    pub subkey: u16,
}

#[derive(Error, Debug)]
pub enum ExtractError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("UTF error: {0}")]
    Utf(#[from] crate::acb::utf::UtfError),
    #[error("Track error: {0}")]
    Track(#[from] crate::acb::track::TrackError),
    #[error("AFS error: {0}")]
    Afs(#[from] crate::acb::afs::AfsError),
    #[error("Invalid ACB file")]
    InvalidAcb,
}

/// Extract all audio files from an ACB file
pub fn extract_acb<R: Read + Seek>(
    acb_file: R,
    target_dir: &Path,
    acb_file_path: Option<&Path>,
) -> Result<Vec<String>, ExtractError> {
    let mut utf = UtfTable::new(acb_file)?;

    let track_list = TrackList::new(&utf)?;

    let mut embedded_awb = load_embedded_awb(&mut utf.rows[0]);
    let mut external_awbs = load_external_awbs(&utf.rows[0], acb_file_path);

    extract_all_tracks(
        &track_list,
        target_dir,
        &mut embedded_awb,
        &mut external_awbs,
    )
}

/// Extract all audio tracks from an ACB reader into memory.
pub fn extract_acb_to_memory<R: Read + Seek>(
    acb_file: R,
    acb_file_path: Option<&Path>,
) -> Result<Vec<ExtractedAcbTrack>, ExtractError> {
    let mut utf = UtfTable::new(acb_file)?;
    let track_list = TrackList::new(&utf)?;
    let mut embedded_awb = load_embedded_awb(&mut utf.rows[0]);
    let mut external_awbs = load_external_awbs(&utf.rows[0], acb_file_path);

    let mut outputs = Vec::new();
    for track in &track_list.tracks {
        let (data, subkey) = match get_track_data(track, &mut embedded_awb, &mut external_awbs)? {
            Some(data) => data,
            None => continue,
        };
        outputs.push(ExtractedAcbTrack {
            name: track.name.clone(),
            cue_id: track.cue_id,
            extension: track_extension(track),
            data,
            subkey,
        });
    }

    Ok(outputs)
}

/// A cue that references a waveform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcbCueRef {
    pub name: String,
    pub cue_id: i32,
}

/// A distinct (de-duplicated) waveform from an ACB, with every cue that maps to
/// it. ACBs frequently point multiple cues at one physical waveform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniqueWaveform {
    pub extension: String,
    /// AFS2 subkey of the originating AWB (0 if unencrypted).
    pub subkey: u16,
    pub data: Vec<u8>,
    /// All cues that reference this waveform (always at least one).
    pub cues: Vec<AcbCueRef>,
}

/// Extract each distinct waveform from an ACB exactly once, together with the
/// cues that reference it. The per-cue extractors read and copy a shared
/// waveform once per cue; this reads and copies each waveform a single time.
pub fn extract_acb_unique_to_memory<R: Read + Seek>(
    acb_file: R,
    acb_file_path: Option<&Path>,
) -> Result<Vec<UniqueWaveform>, ExtractError> {
    let mut utf = UtfTable::new(acb_file)?;
    let track_list = TrackList::new(&utf)?;
    let mut embedded_awb = load_embedded_awb(&mut utf.rows[0]);
    let mut external_awbs = load_external_awbs(&utf.rows[0], acb_file_path);

    // A physical waveform is identified by which AWB it lives in plus its id.
    let mut seen: HashMap<(bool, i32, i32), usize> = HashMap::new();
    let mut out: Vec<UniqueWaveform> = Vec::new();

    for track in &track_list.tracks {
        let key = (track.is_stream, track.stream_awb_id, track.wav_id);
        let cue = AcbCueRef {
            name: track.name.clone(),
            cue_id: track.cue_id,
        };
        if let Some(&idx) = seen.get(&key) {
            out[idx].cues.push(cue);
            continue;
        }
        let (data, subkey) = match get_track_data(track, &mut embedded_awb, &mut external_awbs)? {
            Some(d) => d,
            None => continue,
        };
        seen.insert(key, out.len());
        out.push(UniqueWaveform {
            extension: track_extension(track),
            subkey,
            data,
            cues: vec![cue],
        });
    }

    Ok(out)
}

/// Output extension for a track's waveform (no leading dot; falls back to the
/// numeric encode type for unknown formats).
fn track_extension(track: &Track) -> String {
    let ext = wave_type_extension(track.enc_type);
    if ext.is_empty() {
        track.enc_type.to_string()
    } else {
        ext.trim_start_matches('.').to_string()
    }
}

fn load_embedded_awb(row: &mut crate::acb::utf::ValueMap) -> Option<AfsArchive<Cursor<Vec<u8>>>> {
    // Move the embedded AWB bytes out of the UTF cell rather than cloning them
    // (the embedded AWB is often several MB).
    let awb_data = take_bytes_field(row, "AwbFile")?;
    if awb_data.is_empty() {
        return None;
    }
    AfsArchive::new(Cursor::new(awb_data)).ok()
}

fn load_external_awbs(
    row: &crate::acb::utf::ValueMap,
    acb_file_path: Option<&Path>,
) -> Vec<AfsArchive<Cursor<Vec<u8>>>> {
    let mut external_awbs = Vec::new();

    let stream_awb_hash = match get_bytes_field(row, "StreamAwbHash") {
        Some(data) if !data.is_empty() => data,
        _ => return external_awbs,
    };

    let hash_table = match UtfTable::new(Cursor::new(stream_awb_hash)) {
        Ok(t) => t,
        Err(_) => return external_awbs,
    };

    let acb_dir = acb_file_path.and_then(Path::parent);

    for awb_row in &hash_table.rows {
        let awb_name = match get_string_field(awb_row, "Name") {
            Some(n) => n,
            None => continue,
        };

        let awb_path = match acb_dir {
            Some(dir) => dir.join(format!("{}.awb", awb_name)),
            None => continue,
        };

        if let Some(awb) = load_external_awb_file(&awb_path) {
            external_awbs.push(awb);
        }
    }

    external_awbs
}

fn load_external_awb_file(awb_path: &Path) -> Option<AfsArchive<Cursor<Vec<u8>>>> {
    if !awb_path.exists() {
        return None;
    }

    let awb_data = fs::read(awb_path).ok()?;
    AfsArchive::new(Cursor::new(awb_data)).ok()
}

fn extract_all_tracks(
    track_list: &TrackList,
    target_dir: &Path,
    embedded_awb: &mut Option<AfsArchive<Cursor<Vec<u8>>>>,
    external_awbs: &mut [AfsArchive<Cursor<Vec<u8>>>],
) -> Result<Vec<String>, ExtractError> {
    let mut outputs = Vec::new();
    fs::create_dir_all(target_dir)?;

    for track in &track_list.tracks {
        if let Some(output_path) =
            extract_single_track(track, target_dir, embedded_awb, external_awbs)?
        {
            outputs.push(output_path);
        }
    }

    Ok(outputs)
}

fn extract_single_track(
    track: &Track,
    target_dir: &Path,
    embedded_awb: &mut Option<AfsArchive<Cursor<Vec<u8>>>>,
    external_awbs: &mut [AfsArchive<Cursor<Vec<u8>>>],
) -> Result<Option<String>, ExtractError> {
    let ext = wave_type_extension(track.enc_type);
    let ext = if ext.is_empty() {
        format!(".{}", track.enc_type)
    } else {
        ext.to_string()
    };

    let filename = format!("{}{}", track.name, ext);
    let output_path = target_dir.join(&filename);

    let data = get_track_data(track, embedded_awb, external_awbs)?;
    let (data, _subkey) = match data {
        Some(d) => d,
        None => return Ok(None),
    };

    fs::write(&output_path, data)?;
    Ok(Some(output_path.to_string_lossy().into_owned()))
}

/// Returns the raw waveform bytes plus the originating AWB's AFS2 subkey (for
/// type-56 HCA decryption; 0 if the AWB is unencrypted).
fn get_track_data(
    track: &Track,
    embedded_awb: &mut Option<AfsArchive<Cursor<Vec<u8>>>>,
    external_awbs: &mut [AfsArchive<Cursor<Vec<u8>>>],
) -> Result<Option<(Vec<u8>, u16)>, ExtractError> {
    if track.is_stream {
        if track.stream_awb_id >= 0 && (track.stream_awb_id as usize) < external_awbs.len() {
            let awb = &mut external_awbs[track.stream_awb_id as usize];
            let data = awb.file_data_for_cue_id(track.wav_id)?;
            return Ok(Some((data, awb.subkey)));
        }
    } else if let Some(awb) = embedded_awb.as_mut() {
        let data = awb.file_data_for_cue_id(track.wav_id)?;
        return Ok(Some((data, awb.subkey)));
    }

    Ok(None)
}

/// A track extracted to disk together with the metadata needed to decode it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedTrackFile {
    /// Path of the written waveform file.
    pub path: String,
    /// Cue name of the track (also the file stem).
    pub name: String,
    /// Cue id of the track.
    pub cue_id: i32,
    /// AFS2 subkey of the originating AWB. Required (with the global keycode) to
    /// decrypt type-56 encrypted HCA; 0 when the AWB is unencrypted.
    pub subkey: u16,
}

/// Extract all audio tracks to `target_dir`, returning per-track metadata
/// (output path, cue name/id, and the originating AWB's AFS2 subkey).
pub fn extract_acb_tracks<R: Read + Seek>(
    acb_file: R,
    target_dir: &Path,
    acb_file_path: Option<&Path>,
) -> Result<Vec<ExtractedTrackFile>, ExtractError> {
    let mut utf = UtfTable::new(acb_file)?;
    let track_list = TrackList::new(&utf)?;
    let mut embedded_awb = load_embedded_awb(&mut utf.rows[0]);
    let mut external_awbs = load_external_awbs(&utf.rows[0], acb_file_path);

    fs::create_dir_all(target_dir)?;

    let mut outputs = Vec::new();
    for track in &track_list.tracks {
        if let Some(info) =
            extract_single_track_file(track, target_dir, &mut embedded_awb, &mut external_awbs)?
        {
            outputs.push(info);
        }
    }
    Ok(outputs)
}

fn extract_single_track_file(
    track: &Track,
    target_dir: &Path,
    embedded_awb: &mut Option<AfsArchive<Cursor<Vec<u8>>>>,
    external_awbs: &mut [AfsArchive<Cursor<Vec<u8>>>],
) -> Result<Option<ExtractedTrackFile>, ExtractError> {
    let ext = wave_type_extension(track.enc_type);
    let ext = if ext.is_empty() {
        format!(".{}", track.enc_type)
    } else {
        ext.to_string()
    };

    let filename = format!("{}{}", track.name, ext);
    let output_path = target_dir.join(&filename);

    let (data, subkey) = match get_track_data(track, embedded_awb, external_awbs)? {
        Some(d) => d,
        None => return Ok(None),
    };

    fs::write(&output_path, data)?;
    Ok(Some(ExtractedTrackFile {
        path: output_path.to_string_lossy().into_owned(),
        name: track.name.clone(),
        cue_id: track.cue_id,
        subkey,
    }))
}

/// Read and validate an ACB file into memory, returning its bytes or `None` if
/// the path is missing or is not a valid ACB.
///
/// Slurping the whole file once (instead of parsing straight from the `File`)
/// turns the parser's many small `seek`/`read` calls into in-memory pointer
/// math rather than syscalls — a large win on the file-based entry points.
pub fn read_validated_acb(acb_path: &Path) -> Result<Option<Vec<u8>>, ExtractError> {
    let data = match fs::read(acb_path) {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };

    // A valid ACB file must have at least @UTF magic (4 bytes) + header (28 bytes) = 32 bytes,
    // and start with the @UTF magic (0x40 0x55 0x54 0x46).
    if data.len() < 32 || data[0..4] != [0x40, 0x55, 0x54, 0x46] {
        return Ok(None);
    }

    Ok(Some(data))
}

/// Convenience function to extract from a file path
pub fn extract_acb_from_file(
    acb_path: &Path,
    target_dir: &Path,
) -> Result<Option<Vec<String>>, ExtractError> {
    let data = match read_validated_acb(acb_path)? {
        Some(d) => d,
        None => return Ok(None),
    };
    let outputs = extract_acb(Cursor::new(data), target_dir, Some(acb_path))?;
    Ok(Some(outputs))
}

/// Like [`extract_acb_from_file`], but returns per-track metadata (output path,
/// cue name/id, and AFS2 subkey) instead of just the written paths.
pub fn extract_acb_tracks_from_file(
    acb_path: &Path,
    target_dir: &Path,
) -> Result<Option<Vec<ExtractedTrackFile>>, ExtractError> {
    let data = match read_validated_acb(acb_path)? {
        Some(d) => d,
        None => return Ok(None),
    };
    let outputs = extract_acb_tracks(Cursor::new(data), target_dir, Some(acb_path))?;
    Ok(Some(outputs))
}
