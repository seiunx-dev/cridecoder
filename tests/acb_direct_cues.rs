//! Direct waveform cues need no track/event graph. Build fixtures in memory so
//! these regressions do not depend on proprietary game assets or audio decoding.

use cridecoder::acb::{
    ColumnDef, Value, COLUMN_TYPE_1BYTE, COLUMN_TYPE_2BYTE, COLUMN_TYPE_DATA, COLUMN_TYPE_STRING,
};
use cridecoder::{extract_acb_to_memory, AfsArchiveBuilder, ExtractedAcbTrack, UtfTableBuilder};
use std::collections::HashMap;
use std::io::Cursor;

const WAVEFORMS: [(u16, &[u8]); 2] = [(7, b"first waveform"), (23, b"second waveform")];
const SUBKEY: u16 = 0x1234;

fn utf_table(name: &str, columns: &[(&str, u8)], rows: Vec<Vec<Value>>) -> Vec<u8> {
    let mut builder = UtfTableBuilder::new(name);
    for &(name, typ) in columns {
        builder.add_column(ColumnDef::per_row(name, typ));
    }
    for values in rows {
        assert_eq!(columns.len(), values.len());
        builder.add_row(
            columns
                .iter()
                .zip(values)
                .map(|(&(name, _), value)| (name.to_string(), value))
                .collect(),
        );
    }
    let mut output = Cursor::new(Vec::new());
    builder.build(&mut output).unwrap();
    output.into_inner()
}

#[derive(Clone, Copy)]
enum EmptyTables {
    Bytes,
    Utf,
}

fn direct_acb(cues: &[(&str, u16)], empty_tables: EmptyTables) -> Vec<u8> {
    let cue_table = utf_table(
        "CueTable",
        &[
            ("ReferenceType", COLUMN_TYPE_1BYTE),
            ("ReferenceIndex", COLUMN_TYPE_2BYTE),
        ],
        cues.iter()
            .map(|&(_, index)| vec![Value::U8(1), Value::U16(index)])
            .collect(),
    );
    let name_table = utf_table(
        "CueNameTable",
        &[
            ("CueIndex", COLUMN_TYPE_2BYTE),
            ("CueName", COLUMN_TYPE_STRING),
        ],
        // Names bind by CueIndex even when table rows are in a different order.
        cues.iter()
            .enumerate()
            .rev()
            .map(|(index, &(name, _))| vec![Value::U16(index as u16), Value::String(name.into())])
            .collect(),
    );
    let waveform_table = utf_table(
        "WaveformTable",
        &[
            ("Streaming", COLUMN_TYPE_1BYTE),
            ("EncodeType", COLUMN_TYPE_1BYTE),
            ("MemoryAwbId", COLUMN_TYPE_2BYTE),
        ],
        WAVEFORMS
            .iter()
            .map(|&(id, _)| vec![Value::U8(0), Value::U8(0), Value::U16(id)])
            .collect(),
    );
    let (tracks, events) = match empty_tables {
        EmptyTables::Bytes => (Vec::new(), Vec::new()),
        EmptyTables::Utf => (
            utf_table("TrackTable", &[("EventIndex", COLUMN_TYPE_2BYTE)], vec![]),
            utf_table("TrackEventTable", &[("Command", COLUMN_TYPE_DATA)], vec![]),
        ),
    };
    let mut awb = AfsArchiveBuilder::new()
        .with_alignment(1)
        .with_subkey(SUBKEY);
    for &(id, bytes) in &WAVEFORMS {
        awb.add_file(u32::from(id), bytes.to_vec());
    }
    let mut awb_data = Cursor::new(Vec::new());
    awb.build(&mut awb_data).unwrap();

    let mut header = UtfTableBuilder::new("Header");
    for (name, data) in [
        ("CueTable", cue_table),
        ("CueNameTable", name_table),
        ("WaveformTable", waveform_table),
        ("TrackTable", tracks),
        ("TrackEventTable", events),
        ("AwbFile", awb_data.into_inner()),
    ] {
        header.add_column(ColumnDef::constant(name, Value::Data(data)));
    }
    header.add_row(HashMap::new());
    let mut output = Cursor::new(Vec::new());
    header.build(&mut output).unwrap();
    output.into_inner()
}

fn expected_track(name: &str, cue_id: i32, waveform_index: usize) -> ExtractedAcbTrack {
    ExtractedAcbTrack {
        name: name.into(),
        cue_id,
        extension: "adx".into(),
        data: WAVEFORMS[waveform_index].1.to_vec(),
        subkey: SUBKEY,
    }
}

#[test]
fn extracts_direct_cues_with_empty_track_and_event_tables() {
    for tables in [EmptyTables::Bytes, EmptyTables::Utf] {
        let acb = direct_acb(&[("direct_voice", 0)], tables);
        let tracks = extract_acb_to_memory(Cursor::new(acb), None).unwrap();
        assert_eq!(tracks, vec![expected_track("direct_voice", 0, 0)]);
    }
}

#[test]
fn direct_cues_keep_cue_names_and_ids_when_waveform_indices_differ() {
    let acb = direct_acb(
        &[("second", 1), ("first", 0), ("shared", 1)],
        EmptyTables::Bytes,
    );
    let tracks = extract_acb_to_memory(Cursor::new(acb), None).unwrap();
    assert_eq!(
        tracks,
        vec![
            expected_track("second", 0, 1),
            expected_track("first", 1, 0),
            expected_track("shared", 2, 1),
        ]
    );
}

#[test]
fn skips_direct_cues_with_out_of_range_waveform_indices() {
    let acb = direct_acb(&[("missing", u16::MAX), ("valid", 0)], EmptyTables::Bytes);
    let tracks = extract_acb_to_memory(Cursor::new(acb), None).unwrap();
    assert_eq!(tracks, vec![expected_track("valid", 1, 0)]);
}
