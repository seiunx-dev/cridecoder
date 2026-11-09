//! Track list extraction from ACB

use crate::acb::utf::{get_bytes_field, get_int_field, get_string_field, UtfTable, ValueMap};
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TrackError {
    #[error("UTF error: {0}")]
    Utf(#[from] crate::acb::utf::UtfError),
    #[error("No rows in UTF table")]
    NoRows,
    #[error("Missing required table: {0}")]
    MissingTable(String),
    #[error("ReferenceType {0} not implemented")]
    UnsupportedRefType(i32),
}

/// Represents an audio track
#[derive(Debug, Clone)]
pub struct Track {
    pub cue_id: i32,
    pub name: String,
    pub wav_id: i32,
    pub enc_type: i32,
    pub is_stream: bool,
    pub stream_awb_id: i32,
}

/// List of tracks from an ACB
#[derive(Debug, Clone)]
pub struct TrackList {
    pub tracks: Vec<Track>,
}

struct AcbTables {
    cues: UtfTable,
    nams: UtfTable,
    wavs: Option<UtfTable>,
    syns: Option<UtfTable>,
    tras: Option<UtfTable>,
    tevs: Option<UtfTable>,
    seqs: Option<UtfTable>,
}

impl TrackList {
    /// Create a TrackList from a UTF table
    pub fn new(utf: &UtfTable) -> Result<Self, TrackError> {
        if utf.rows.is_empty() {
            return Err(TrackError::NoRows);
        }

        let tables = parse_acb_tables(&utf.rows[0])?;
        let name_map = build_name_map(&tables.nams);

        let mut tl = TrackList { tracks: Vec::new() };
        // Track all chosen names so duplicate detection is O(1) instead of an
        // O(n^2) scan of every prior track name.
        let mut seen_names: HashSet<String> = HashSet::new();
        extract_tracks_from_tables(&tables, &name_map, &mut tl, &mut seen_names)?;

        Ok(tl)
    }
}

fn parse_acb_tables(row: &ValueMap) -> Result<AcbTables, TrackError> {
    let table_bytes = extract_table_bytes(row)?;
    parse_utf_tables(&table_bytes)
}

fn extract_table_bytes(row: &ValueMap) -> Result<HashMap<&str, Vec<u8>>, TrackError> {
    let mut tables = HashMap::new();

    // Required tables
    for name in &["CueTable", "CueNameTable", "TrackTable"] {
        let data =
            get_bytes_field(row, name).ok_or_else(|| TrackError::MissingTable(name.to_string()))?;
        tables.insert(*name, data.to_vec());
    }

    // Optional tables
    for name in &["WaveformTable", "SynthTable"] {
        if let Some(data) = get_bytes_field(row, name) {
            if !data.is_empty() {
                tables.insert(*name, data.to_vec());
            }
        }
    }

    // TrackEventTable or CommandTable
    let tev_data = get_bytes_field(row, "TrackEventTable")
        .or_else(|| get_bytes_field(row, "CommandTable"))
        .ok_or_else(|| TrackError::MissingTable("TrackEventTable/CommandTable".to_string()))?;
    tables.insert("TrackEventTable", tev_data.to_vec());

    // Optional SequenceTable
    if let Some(data) = get_bytes_field(row, "SequenceTable") {
        if !data.is_empty() {
            tables.insert("SequenceTable", data.to_vec());
        }
    }

    Ok(tables)
}

fn parse_utf_tables(table_bytes: &HashMap<&str, Vec<u8>>) -> Result<AcbTables, TrackError> {
    let cues = UtfTable::new(Cursor::new(&table_bytes["CueTable"]))?;
    let nams = UtfTable::new(Cursor::new(&table_bytes["CueNameTable"]))?;

    let wavs = table_bytes
        .get("WaveformTable")
        .filter(|d| !d.is_empty())
        .map(|d| UtfTable::new(Cursor::new(d)))
        .transpose()?;

    let syns = table_bytes
        .get("SynthTable")
        .filter(|d| !d.is_empty())
        .map(|d| UtfTable::new(Cursor::new(d)))
        .transpose()?;

    let tras = table_bytes
        .get("TrackTable")
        .filter(|d| !d.is_empty())
        .map(|d| UtfTable::new(Cursor::new(d)))
        .transpose()?;
    let tevs = table_bytes
        .get("TrackEventTable")
        .filter(|d| !d.is_empty())
        .map(|d| UtfTable::new(Cursor::new(d)))
        .transpose()?;

    let seqs = table_bytes
        .get("SequenceTable")
        .filter(|d| !d.is_empty())
        .map(|d| UtfTable::new(Cursor::new(d)))
        .transpose()?;

    Ok(AcbTables {
        cues,
        nams,
        wavs,
        syns,
        tras,
        tevs,
        seqs,
    })
}

fn build_name_map(nams: &UtfTable) -> HashMap<i32, String> {
    let mut name_map = HashMap::new();
    for row in &nams.rows {
        let idx = get_int_field(row, "CueIndex") as i32;
        let name = get_string_field(row, "CueName").unwrap_or("").to_string();
        name_map.insert(idx, name);
    }
    name_map
}

fn extract_tracks_from_tables(
    tables: &AcbTables,
    name_map: &HashMap<i32, String>,
    tl: &mut TrackList,
    seen: &mut HashSet<String>,
) -> Result<(), TrackError> {
    for (cue_index, cue_row) in tables.cues.rows.iter().enumerate() {
        let ref_type = get_int_field(cue_row, "ReferenceType") as i32;
        let cue_index = cue_index as i32;
        let ref_index = get_int_field(cue_row, "ReferenceIndex") as usize;

        if ref_type == 1 {
            if let Some(track) = extract_track_from_waveform(
                tables.wavs.as_ref(),
                name_map,
                cue_index,
                ref_index,
                seen,
            ) {
                tl.tracks.push(track);
            }
            continue;
        }

        // vgmstream skips cues with an unhandled ReferenceType rather than failing
        // the whole ACB (acb.c). Track graphs use 3 (Sequence) or 8 (BlockSequence).
        if ref_type != 3 && ref_type != 8 {
            continue;
        }

        // The cue NAME binds to the cue's row position (CueIndex); ReferenceIndex
        // only indexes the Sequence/Synth tables for descent. They differ when
        // CueTable order != SequenceTable order, so thread both (vgmstream acb.c).
        if let Some(seqs) = &tables.seqs {
            if ref_index < seqs.rows.len() {
                extract_sequence_tracks(tables, name_map, ref_index, cue_index, tl, seen);
                continue;
            }
        }

        extract_direct_tracks(tables, name_map, cue_index, tl, seen);
    }

    Ok(())
}

fn extract_sequence_tracks(
    tables: &AcbTables,
    name_map: &HashMap<i32, String>,
    ref_index: usize,
    cue_index: i32,
    tl: &mut TrackList,
    seen: &mut HashSet<String>,
) {
    let Some(tras) = &tables.tras else {
        return;
    };

    if let Some(seqs) = &tables.seqs {
        let seq = &seqs.rows[ref_index];
        let num_tracks = get_int_field(seq, "NumTracks") as usize;

        if let Some(track_index_data) = get_bytes_field(seq, "TrackIndex") {
            for i in 0..num_tracks {
                if i * 2 + 1 >= track_index_data.len() {
                    break;
                }
                let idx = u16::from_be_bytes([track_index_data[i * 2], track_index_data[i * 2 + 1]])
                    as usize;
                if idx < tras.rows.len() {
                    extract_track_from_track_row(tables, name_map, cue_index, idx, tl, seen);
                }
            }
        }
    }
}

fn extract_direct_tracks(
    tables: &AcbTables,
    name_map: &HashMap<i32, String>,
    cue_index: i32,
    tl: &mut TrackList,
    seen: &mut HashSet<String>,
) {
    let Some(tras) = &tables.tras else {
        return;
    };

    for idx in 0..tras.rows.len() {
        extract_track_from_track_row(tables, name_map, cue_index, idx, tl, seen);
    }
}

fn extract_track_from_track_row(
    tables: &AcbTables,
    name_map: &HashMap<i32, String>,
    cue_index: i32,
    track_idx: usize,
    tl: &mut TrackList,
    seen: &mut HashSet<String>,
) {
    let (Some(tras), Some(tevs)) = (&tables.tras, &tables.tevs) else {
        return;
    };
    let track = &tras.rows[track_idx];
    let event_idx = get_int_field(track, "EventIndex") as usize;

    if event_idx == 0xFFFF || event_idx >= tevs.rows.len() {
        return;
    }

    let tracks = extract_tracks_from_event(
        &tevs.rows[event_idx],
        tables.syns.as_ref(),
        tables.wavs.as_ref(),
        name_map,
        cue_index,
        seen,
    );

    tl.tracks.extend(tracks);
}

fn extract_tracks_from_event(
    track_event: &ValueMap,
    syns: Option<&UtfTable>,
    wavs: Option<&UtfTable>,
    name_map: &HashMap<i32, String>,
    cue_index: i32,
    seen: &mut HashSet<String>,
) -> Vec<Track> {
    let mut tracks = Vec::new();

    let command = match get_bytes_field(track_event, "Command") {
        Some(c) => c,
        None => return tracks,
    };

    let mut k = 0;
    while k < command.len() {
        if k + 3 > command.len() {
            break;
        }

        let cmd = u16::from_be_bytes([command[k], command[k + 1]]);
        let cmd_len = command[k + 2] as usize;
        k += 3;

        if k + cmd_len > command.len() {
            break;
        }

        let param_bytes = &command[k..k + cmd_len];
        k += cmd_len;

        if cmd == 0 {
            break;
        }

        if cmd == 0x07d0 {
            if let Some(track) =
                extract_track_from_command(param_bytes, syns, wavs, name_map, cue_index, seen)
            {
                tracks.push(track);
            }
        }
    }

    tracks
}

fn extract_track_from_command(
    param_bytes: &[u8],
    syns: Option<&UtfTable>,
    wavs: Option<&UtfTable>,
    name_map: &HashMap<i32, String>,
    cue_index: i32,
    seen: &mut HashSet<String>,
) -> Option<Track> {
    if param_bytes.len() < 4 {
        return None;
    }

    let u1 = u16::from_be_bytes([param_bytes[0], param_bytes[1]]);
    if u1 != 2 {
        return None;
    }

    let syn_idx = u16::from_be_bytes([param_bytes[2], param_bytes[3]]) as usize;
    let syns = syns?;
    if syn_idx >= syns.rows.len() {
        return None;
    }

    let r_data = get_bytes_field(&syns.rows[syn_idx], "ReferenceItems")?;
    if r_data.len() < 4 {
        return None;
    }

    let a = u16::from_be_bytes([r_data[0], r_data[1]]);
    let wav_idx = u16::from_be_bytes([r_data[2], r_data[3]]) as usize;

    if a != 1 {
        return None;
    }

    extract_track_from_waveform(wavs, name_map, cue_index, wav_idx, seen)
}

fn extract_track_from_waveform(
    wavs: Option<&UtfTable>,
    name_map: &HashMap<i32, String>,
    cue_index: i32,
    wav_idx: usize,
    seen: &mut HashSet<String>,
) -> Option<Track> {
    let wavs = wavs?;
    let wav_row = wavs.rows.get(wav_idx)?;
    let is_stream = get_int_field(wav_row, "Streaming") != 0;
    let enc_type = get_int_field(wav_row, "EncodeType") as i32;

    let wav_id = if is_stream {
        get_int_field(wav_row, "StreamAwbId") as i32
    } else {
        get_int_field(wav_row, "MemoryAwbId") as i32
    };

    let stream_awb_id = if is_stream {
        get_int_field(wav_row, "StreamAwbPortNo") as i32
    } else {
        -1
    };

    let base_name = name_map.get(&cue_index).cloned().unwrap_or_default();
    let name = generate_unique_name(&base_name, cue_index, wav_id, seen);

    Some(Track {
        cue_id: cue_index,
        name,
        wav_id,
        enc_type,
        is_stream,
        stream_awb_id,
    })
}

fn generate_unique_name(
    base_name: &str,
    ref_index: i32,
    wav_id: i32,
    seen: &mut HashSet<String>,
) -> String {
    let name = if base_name.is_empty() {
        format!("UNKNOWN-{}", ref_index)
    } else {
        base_name.to_string()
    };

    // O(1) duplicate check against all previously chosen names. Matches the
    // prior behavior: a clashing base name gets a `-{wav_id}` suffix.
    let final_name = if seen.contains(&name) {
        format!("{}-{}", name, wav_id)
    } else {
        name
    };

    seen.insert(final_name.clone());
    final_name
}
