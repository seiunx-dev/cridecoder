//! USM metadata reading and export

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::reader::Reader;
use encoding_rs::SHIFT_JIS;
use serde::{Serialize, Serializer};
use thiserror::Error;

use super::extractor::{UsmError, UtfRow, UtfTable, UtfValue};

/// Metadata for a USM file
#[derive(Debug, Clone, Serialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_filename: Option<String>,
    pub has_audio: bool,
    pub stream_offset: i64,
    pub sections: Vec<MetadataSection>,
}

/// A section in the USM metadata
#[derive(Debug, Clone, Serialize)]
pub struct MetadataSection {
    pub kind: String,
    pub signature: String,
    pub offset: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<SectionData>,
}

/// Section data can be a table or a marker string
#[derive(Debug, Clone)]
pub enum SectionData {
    Table(TableData),
    Marker(String),
}

impl Serialize for SectionData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            SectionData::Table(table) => table.serialize(serializer),
            SectionData::Marker(marker) => serializer.serialize_str(marker),
        }
    }
}

/// Table data with normalized values
#[derive(Debug, Clone, Serialize)]
pub struct TableData {
    pub table_name: String,
    pub row_count: usize,
    pub rows: Vec<HashMap<String, MetadataValue>>,
}

/// A normalized metadata value
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum MetadataValue {
    Byte(u8),
    SByte(i8),
    UShort(u16),
    Short(i16),
    UInt(u32),
    Int(i32),
    ULong(u64),
    Float(f32),
    String(String),
    Binary(BinarySummary),
}

/// Summary of binary data
#[derive(Debug, Clone, Serialize)]
pub struct BinarySummary {
    pub size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
}

/// Metadata reading errors
#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("USM error: {0}")]
    Usm(#[from] UsmError),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("invalid CRID signature")]
    InvalidCridSignature,
    #[error("invalid UTF signature")]
    InvalidUtfSignature,
    #[error("expected {0} signature")]
    ExpectedSignature(String),
    #[error("expected {0}")]
    ExpectedMarker(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Column storage masks
const COLUMN_STORAGE_MASK: u8 = 0xF0;
const COLUMN_STORAGE_CONSTANT: u8 = 0x30;
const COLUMN_STORAGE_CONSTANT2: u8 = 0x70;

/// Column type masks
const COLUMN_TYPE_MASK: u8 = 0x0F;
const COLUMN_TYPE_DATA: u8 = 0x0B;
const COLUMN_TYPE_STRING: u8 = 0x0A;
const COLUMN_TYPE_FLOAT: u8 = 0x08;
const COLUMN_TYPE_8BYTE: u8 = 0x06;
const COLUMN_TYPE_4BYTE2: u8 = 0x05;
const COLUMN_TYPE_4BYTE: u8 = 0x04;
const COLUMN_TYPE_2BYTE2: u8 = 0x03;
const COLUMN_TYPE_2BYTE: u8 = 0x02;
const COLUMN_TYPE_1BYTE2: u8 = 0x01;
const COLUMN_TYPE_1BYTE: u8 = 0x00;

/// Read null-terminated C string as bytes
fn read_cstring<R: Read + Seek>(reader: &mut Reader<R>) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    loop {
        let b = reader.read_u8()?;
        if b == 0 {
            break;
        }
        buf.push(b);
    }
    Ok(buf)
}

/// Align stream to boundary
fn align_stream<R: Read + Seek>(reader: &mut Reader<R>, alignment: u64) -> io::Result<u64> {
    let pos = reader.stream_position()?;
    let remainder = pos % alignment;
    if remainder != 0 {
        reader.seek(SeekFrom::Current((alignment - remainder) as i64))
    } else {
        Ok(pos)
    }
}

/// Detailed UTF table with name
struct DetailedUtfTable {
    name: String,
    rows: UtfTable,
}

/// Field info for UTF table parsing
struct FieldInfo {
    name: String,
    column_type: u8,
    constant: Option<UtfValue>,
}

/// Read column data from a UTF table
fn read_column_data<R: Read + Seek>(
    reader: &mut Reader<R>,
    column_type: u8,
    string_table_offset: i64,
    data_offset: i64,
) -> Result<UtfValue, MetadataError> {
    match column_type {
        COLUMN_TYPE_DATA => {
            let offset = reader.read_u32()?;
            let size = reader.read_u32()?;
            let current_pos = reader.stream_position()?;
            reader.seek(SeekFrom::Start((data_offset + offset as i64 - 24) as u64))?;
            let data = reader.read_bytes(size as usize)?;
            reader.seek(SeekFrom::Start(current_pos))?;
            Ok(UtfValue::Data(data))
        }
        COLUMN_TYPE_STRING => {
            let offset = reader.read_u32()?;
            let current_pos = reader.stream_position()?;
            reader.seek(SeekFrom::Start(
                (string_table_offset + offset as i64 - 24) as u64,
            ))?;
            let s = read_cstring(reader)?;
            reader.seek(SeekFrom::Start(current_pos))?;
            Ok(UtfValue::String(s))
        }
        COLUMN_TYPE_FLOAT => Ok(UtfValue::Float(reader.read_f32()?)),
        COLUMN_TYPE_8BYTE => Ok(UtfValue::ULong(reader.read_u64()?)),
        COLUMN_TYPE_4BYTE2 => Ok(UtfValue::Int(reader.read_i32()?)),
        COLUMN_TYPE_4BYTE => Ok(UtfValue::UInt(reader.read_u32()?)),
        COLUMN_TYPE_2BYTE2 => Ok(UtfValue::Short(reader.read_i16()?)),
        COLUMN_TYPE_2BYTE => Ok(UtfValue::UShort(reader.read_u16()?)),
        COLUMN_TYPE_1BYTE2 => Ok(UtfValue::SByte(reader.read_i8()?)),
        COLUMN_TYPE_1BYTE => Ok(UtfValue::Byte(reader.read_u8()?)),
        _ => Err(MetadataError::Usm(UsmError::UnknownColumnType(column_type))),
    }
}

/// Parse a detailed UTF table from the reader
fn get_detailed_utf_table<R: Read + Seek>(
    reader: &mut Reader<R>,
) -> Result<DetailedUtfTable, MetadataError> {
    let sig = reader.read_bytes(4)?;
    if &sig != b"@UTF" {
        return Err(MetadataError::InvalidUtfSignature);
    }

    let table_size = reader.read_u32()?;
    let _version = reader.read_u16()?;
    let row_offset = reader.read_u16()?;
    let string_table_offset = reader.read_u32()?;
    let data_offset = reader.read_u32()?;
    let table_name_offset = reader.read_u32()?;
    let number_of_fields = reader.read_u16()?;
    let _row_size = reader.read_u16()?;
    let number_of_rows = reader.read_u32()?;

    let table_data = reader.read_bytes((table_size - 24) as usize)?;
    let mut utf_reader = Reader::new(io::Cursor::new(table_data));

    // Read table name
    utf_reader.seek(SeekFrom::Start(
        (string_table_offset as i64 + table_name_offset as i64 - 24) as u64,
    ))?;
    let table_name_bytes = read_cstring(&mut utf_reader)?;
    let table_name = String::from_utf8_lossy(&table_name_bytes).to_string();
    utf_reader.seek(SeekFrom::Start(0))?;

    let mut fields = Vec::with_capacity(number_of_fields as usize);

    for _ in 0..number_of_fields {
        let field_type = utf_reader.read_u8()?;
        let name_offset = utf_reader.read_u32()?;

        let occurrence = field_type & COLUMN_STORAGE_MASK;
        let type_key = field_type & COLUMN_TYPE_MASK;

        // Read field name
        let current_pos = utf_reader.stream_position()?;
        utf_reader.seek(SeekFrom::Start(
            (string_table_offset as i64 + name_offset as i64 - 24) as u64,
        ))?;
        let field_name_bytes = read_cstring(&mut utf_reader)?;
        let field_name = String::from_utf8_lossy(&field_name_bytes).to_string();
        utf_reader.seek(SeekFrom::Start(current_pos))?;

        if occurrence == COLUMN_STORAGE_CONSTANT || occurrence == COLUMN_STORAGE_CONSTANT2 {
            let field_val = read_column_data(
                &mut utf_reader,
                type_key,
                string_table_offset as i64,
                data_offset as i64,
            )?;
            fields.push(FieldInfo {
                name: field_name,
                column_type: type_key,
                constant: Some(field_val),
            });
        } else {
            fields.push(FieldInfo {
                name: field_name,
                column_type: type_key,
                constant: None,
            });
        }
    }

    utf_reader.seek(SeekFrom::Start((row_offset as i64 - 24) as u64))?;

    let mut rows = Vec::with_capacity(number_of_rows as usize);
    for _ in 0..number_of_rows {
        let mut row = UtfRow::new();
        for field in &fields {
            if let Some(ref constant) = field.constant {
                row.insert(field.name.clone(), constant.clone());
            } else {
                let val = read_column_data(
                    &mut utf_reader,
                    field.column_type,
                    string_table_offset as i64,
                    data_offset as i64,
                )?;
                row.insert(field.name.clone(), val);
            }
        }
        rows.push(row);
    }

    Ok(DetailedUtfTable {
        name: table_name,
        rows,
    })
}

/// Read metadata from a USM reader
pub fn read_metadata<R: Read + Seek>(
    usm: R,
    fallback_name: &[u8],
) -> Result<Metadata, MetadataError> {
    let mut reader = Reader::new(usm);

    let sig = reader.read_bytes(4)?;
    if &sig != b"CRID" {
        return Err(MetadataError::InvalidCridSignature);
    }

    let block_size = reader.read_u32()?;

    // Read CRID table
    reader.seek(SeekFrom::Start(0x20))?;
    let crid_table = get_detailed_utf_table(&mut reader)?;
    let container_filename = extract_container_filename(&crid_table.rows, fallback_name);

    let mut sections = vec![MetadataSection {
        kind: "crid".to_string(),
        signature: "CRID".to_string(),
        offset: 0,
        block_size: Some(block_size),
        data: Some(SectionData::Table(normalize_detailed_utf_table(
            &crid_table,
        ))),
    }];

    let offset = 8 + block_size as i64;
    let (has_audio, mut metadata_sections, stream_offset) =
        read_metadata_sections(&mut reader, offset)?;
    sections.append(&mut metadata_sections);

    Ok(Metadata {
        input_file: None,
        container_filename: Some(container_filename),
        has_audio,
        stream_offset,
        sections,
    })
}

fn read_metadata_sections<R: Read + Seek>(
    reader: &mut Reader<R>,
    mut offset: i64,
) -> Result<(bool, Vec<MetadataSection>, i64), MetadataError> {
    let mut sections = Vec::with_capacity(6);

    // Video header
    let (video_header, next_offset) = read_utf_section(reader, offset, "@SFV", "video_header")?;
    sections.push(video_header);
    offset = next_offset;

    // Check for optional @SFA chunk
    let next_sig = read_signature_at(reader, offset)?;
    let mut has_audio = false;

    if next_sig == "@SFA" {
        let (audio_header, next_offset) = read_utf_section(reader, offset, "@SFA", "audio_header")?;
        sections.push(audio_header);
        offset = next_offset;
        has_audio = true;
    }

    let next_sig = read_signature_at(reader, offset)?;
    if next_sig != "@SFV" {
        return Err(MetadataError::ExpectedSignature("@SFV".to_string()));
    }

    // Video header end
    let (video_header_end, next_offset) =
        read_marker_section(reader, offset, "@SFV", "video_header_end", "#HEADER END")?;
    sections.push(video_header_end);
    offset = next_offset;

    if has_audio {
        let (audio_header_end, next_offset) =
            read_marker_section(reader, offset, "@SFA", "audio_header_end", "#HEADER END")?;
        sections.push(audio_header_end);
        offset = next_offset;
    }

    // Video metadata
    let (video_metadata, next_offset) = read_utf_section(reader, offset, "@SFV", "video_metadata")?;
    sections.push(video_metadata);
    offset = next_offset;

    // Video metadata end
    let (video_metadata_end, stream_offset) = read_metadata_end_section(reader, offset)?;
    sections.push(video_metadata_end);

    Ok((has_audio, sections, stream_offset))
}

fn read_utf_section<R: Read + Seek>(
    reader: &mut Reader<R>,
    offset: i64,
    expected_signature: &str,
    kind: &str,
) -> Result<(MetadataSection, i64), MetadataError> {
    seek_and_check_signature(reader, offset, expected_signature)?;
    let block_size = reader.read_u32()?;

    reader.seek(SeekFrom::Start((offset + 0x20) as u64))?;
    let table = get_detailed_utf_table(reader)?;

    Ok((
        MetadataSection {
            kind: kind.to_string(),
            signature: expected_signature.to_string(),
            offset,
            block_size: Some(block_size),
            data: Some(SectionData::Table(normalize_detailed_utf_table(&table))),
        },
        offset + 8 + block_size as i64,
    ))
}

fn read_marker_section<R: Read + Seek>(
    reader: &mut Reader<R>,
    offset: i64,
    expected_signature: &str,
    kind: &str,
    marker: &str,
) -> Result<(MetadataSection, i64), MetadataError> {
    seek_and_check_signature(reader, offset, expected_signature)?;
    let block_size = reader.read_u32()?;

    reader.seek(SeekFrom::Start((offset + 0x20) as u64))?;
    let value = reader.read_bytes(marker.len())?;

    if value != marker.as_bytes() {
        return Err(MetadataError::ExpectedMarker(marker.to_string()));
    }

    Ok((
        MetadataSection {
            kind: kind.to_string(),
            signature: expected_signature.to_string(),
            offset,
            block_size: Some(block_size),
            data: Some(SectionData::Marker(marker.to_string())),
        },
        offset + 8 + block_size as i64,
    ))
}

fn read_metadata_end_section<R: Read + Seek>(
    reader: &mut Reader<R>,
    offset: i64,
) -> Result<(MetadataSection, i64), MetadataError> {
    seek_and_check_signature(reader, offset, "@SFV")?;
    let block_size = reader.read_u32()?;

    reader.seek(SeekFrom::Start((offset + 0x20) as u64))?;

    let marker = "#METADATA END";
    let value = reader.read_bytes(marker.len())?;

    if value != marker.as_bytes() {
        return Err(MetadataError::ExpectedMarker(marker.to_string()));
    }

    align_stream(reader, 4)?;
    let stream_offset = reader.seek(SeekFrom::Current(16))? as i64;

    Ok((
        MetadataSection {
            kind: "video_metadata_end".to_string(),
            signature: "@SFV".to_string(),
            offset,
            block_size: Some(block_size),
            data: Some(SectionData::Marker(marker.to_string())),
        },
        stream_offset,
    ))
}

fn read_signature_at<R: Read + Seek>(
    reader: &mut Reader<R>,
    offset: i64,
) -> Result<String, MetadataError> {
    reader.seek(SeekFrom::Start(offset as u64))?;
    let sig = reader.read_bytes(4)?;
    Ok(String::from_utf8_lossy(&sig).to_string())
}

fn seek_and_check_signature<R: Read + Seek>(
    reader: &mut Reader<R>,
    offset: i64,
    expected: &str,
) -> Result<(), MetadataError> {
    reader.seek(SeekFrom::Start(offset as u64))?;
    let sig = reader.read_bytes(4)?;
    if sig != expected.as_bytes() {
        return Err(MetadataError::ExpectedSignature(expected.to_string()));
    }
    Ok(())
}

fn extract_container_filename(rows: &UtfTable, fallback_name: &[u8]) -> String {
    if let Some(row) = rows.last() {
        if let Some(value) = row.get("filename") {
            if let Some(text) = stringify_utf_value(value) {
                return text;
            }
        }
    }
    stringify_bytes(fallback_name)
}

fn stringify_utf_value(value: &UtfValue) -> Option<String> {
    match value {
        UtfValue::String(s) => Some(stringify_bytes(s)),
        UtfValue::Data(data) => Some(stringify_bytes(data)),
        _ => None,
    }
}

fn stringify_bytes(data: &[u8]) -> String {
    if let Some(text) = decode_text_bytes(data) {
        text
    } else {
        hex::encode(data)
    }
}

fn decode_text_bytes(data: &[u8]) -> Option<String> {
    if data.is_empty() {
        return Some(String::new());
    }

    // Try UTF-8 first
    if let Ok(text) = std::str::from_utf8(data) {
        if is_mostly_text(text) {
            return Some(text.to_string());
        }
    }

    // Try Shift-JIS
    let (decoded, _, had_errors) = SHIFT_JIS.decode(data);
    if !had_errors && is_mostly_text(&decoded) {
        return Some(decoded.to_string());
    }

    None
}

fn is_mostly_text(s: &str) -> bool {
    for c in s.chars() {
        if c == '\u{FFFD}' {
            return false;
        }
        if c.is_control() && !c.is_whitespace() {
            return false;
        }
    }
    true
}

fn normalize_detailed_utf_table(table: &DetailedUtfTable) -> TableData {
    TableData {
        table_name: table.name.clone(),
        row_count: table.rows.len(),
        rows: normalize_rows(&table.rows),
    }
}

fn normalize_rows(rows: &UtfTable) -> Vec<HashMap<String, MetadataValue>> {
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|(k, v)| (k.clone(), normalize_metadata_value(v)))
                .collect()
        })
        .collect()
}

fn normalize_metadata_value(value: &UtfValue) -> MetadataValue {
    match value {
        UtfValue::Byte(v) => MetadataValue::Byte(*v),
        UtfValue::SByte(v) => MetadataValue::SByte(*v),
        UtfValue::UShort(v) => MetadataValue::UShort(*v),
        UtfValue::Short(v) => MetadataValue::Short(*v),
        UtfValue::UInt(v) => MetadataValue::UInt(*v),
        UtfValue::Int(v) => MetadataValue::Int(*v),
        UtfValue::ULong(v) => MetadataValue::ULong(*v),
        UtfValue::Float(v) => MetadataValue::Float(*v),
        UtfValue::String(data) => {
            if let Some(text) = decode_text_bytes(data) {
                MetadataValue::String(text)
            } else {
                MetadataValue::Binary(summarize_binary(data))
            }
        }
        UtfValue::Data(data) => {
            if let Some(text) = decode_text_bytes(data) {
                MetadataValue::String(text)
            } else {
                MetadataValue::Binary(summarize_binary(data))
            }
        }
    }
}

fn summarize_binary(data: &[u8]) -> BinarySummary {
    const PREVIEW_LIMIT: usize = 32;

    if data.is_empty() {
        return BinarySummary {
            size: 0,
            preview_hex: None,
            truncated: None,
        };
    }

    let preview_size = data.len().min(PREVIEW_LIMIT);
    let truncated = if data.len() > PREVIEW_LIMIT {
        Some(true)
    } else {
        None
    };

    BinarySummary {
        size: data.len(),
        preview_hex: Some(hex::encode(&data[..preview_size])),
        truncated,
    }
}

impl Metadata {
    /// Get video frame rate from metadata
    pub fn video_frame_rate(&self) -> Option<(i32, i32)> {
        for section in &self.sections {
            if section.kind != "video_header" {
                continue;
            }

            if let Some(SectionData::Table(table)) = &section.data {
                if let Some(row) = table.rows.first() {
                    let numerator = metadata_number_to_i32(row.get("framerate_n")?)?;
                    let denominator = metadata_number_to_i32(row.get("framerate_d")?)?;
                    if denominator != 0 {
                        return Some((numerator, denominator));
                    }
                }
            }
        }
        None
    }
}

fn metadata_number_to_i32(value: &MetadataValue) -> Option<i32> {
    match value {
        MetadataValue::Byte(v) => Some(*v as i32),
        MetadataValue::SByte(v) => Some(*v as i32),
        MetadataValue::UShort(v) => Some(*v as i32),
        MetadataValue::Short(v) => Some(*v as i32),
        MetadataValue::UInt(v) => Some(*v as i32),
        MetadataValue::Int(v) => Some(*v),
        MetadataValue::ULong(v) => Some(*v as i32),
        MetadataValue::Float(v) => Some(*v as i32),
        _ => None,
    }
}

/// Read metadata from a file
pub fn read_metadata_file(usm_path: &Path) -> Result<Metadata, MetadataError> {
    let file = File::open(usm_path)?;
    let fallback_name = usm_path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .map(|s| s.as_bytes().to_vec())
        .unwrap_or_default();

    let mut metadata = read_metadata(file, &fallback_name)?;
    metadata.input_file = Some(usm_path.to_string_lossy().to_string());
    Ok(metadata)
}

/// Export metadata to a JSON file
pub fn export_metadata_file(usm_path: &Path, output_path: &Path) -> Result<(), MetadataError> {
    let metadata = read_metadata_file(usm_path)?;
    let mut output_file = File::create(output_path)?;

    let json = serde_json::to_string_pretty(&metadata)?;
    output_file.write_all(json.as_bytes())?;

    Ok(())
}
