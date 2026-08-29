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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usm::extractor::{extract_usm, extract_usm_file, extract_usm_to_memory};
    use std::io::Cursor;

    enum TestField<'a> {
        String(&'a str, &'a str),
        UInt(&'a str, u32),
    }

    fn push_u16(bytes: &mut Vec<u8>, value: u16) {
        bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn push_string(strings: &mut Vec<u8>, value: &str) -> u32 {
        let offset = strings.len() as u32;
        strings.extend_from_slice(value.as_bytes());
        strings.push(0);
        offset
    }

    fn make_utf(table_name: &str, fields: &[TestField<'_>]) -> Vec<u8> {
        let mut strings = Vec::new();
        let table_name_offset = push_string(&mut strings, table_name);
        let mut names = Vec::with_capacity(fields.len());
        let mut string_values = Vec::with_capacity(fields.len());

        for field in fields {
            let (name, value) = match field {
                TestField::String(name, value) => (*name, Some(*value)),
                TestField::UInt(name, _) => (*name, None),
            };
            names.push(push_string(&mut strings, name));
            string_values.push(value.map(|value| push_string(&mut strings, value)));
        }

        let mut schema = Vec::with_capacity(fields.len() * 5);
        let mut row = Vec::with_capacity(fields.len() * 4);
        for ((field, name_offset), string_offset) in fields.iter().zip(names).zip(string_values) {
            match field {
                TestField::String(_, _) => {
                    schema.push(0x50 | COLUMN_TYPE_STRING);
                    push_u32(&mut row, string_offset.expect("string value offset"));
                }
                TestField::UInt(_, value) => {
                    schema.push(0x50 | COLUMN_TYPE_4BYTE);
                    push_u32(&mut row, *value);
                }
            }
            push_u32(&mut schema, name_offset);
        }

        let row_offset = 24 + schema.len();
        let string_table_offset = row_offset + row.len();
        let data_offset = string_table_offset + strings.len();
        let table_size = 24 + schema.len() + row.len() + strings.len();

        let mut table = Vec::with_capacity(8 + table_size);
        table.extend_from_slice(b"@UTF");
        push_u32(&mut table, table_size as u32);
        push_u16(&mut table, 1);
        push_u16(&mut table, row_offset as u16);
        push_u32(&mut table, string_table_offset as u32);
        push_u32(&mut table, data_offset as u32);
        push_u32(&mut table, table_name_offset);
        push_u16(&mut table, fields.len() as u16);
        push_u16(&mut table, row.len() as u16);
        push_u32(&mut table, 1);
        table.extend_from_slice(&schema);
        table.extend_from_slice(&row);
        table.extend_from_slice(&strings);
        table
    }

    fn make_chunk(signature: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut chunk = Vec::with_capacity(32 + payload.len());
        chunk.extend_from_slice(signature);
        push_u32(&mut chunk, (24 + payload.len()) as u32);
        chunk.extend_from_slice(&[0; 24]);
        chunk.extend_from_slice(payload);
        while !chunk.len().is_multiple_of(4) {
            chunk.push(0);
        }
        let block_size = (chunk.len() - 8) as u32;
        chunk[4..8].copy_from_slice(&block_size.to_be_bytes());
        chunk
    }

    fn make_metadata_end_chunk() -> Vec<u8> {
        let mut chunk = make_chunk(b"@SFV", b"#METADATA END");
        chunk.extend_from_slice(&[0; 16]);
        let block_size = (chunk.len() - 8) as u32;
        chunk[4..8].copy_from_slice(&block_size.to_be_bytes());
        chunk
    }

    fn make_stream_chunk(signature: &[u8; 4], payload: &[u8], data_type: u8) -> Vec<u8> {
        let mut chunk = Vec::with_capacity(32 + payload.len());
        chunk.extend_from_slice(signature);
        push_u32(&mut chunk, (24 + payload.len()) as u32);
        push_u16(&mut chunk, 24);
        push_u16(&mut chunk, 0);
        chunk.extend_from_slice(&[0; 3]);
        chunk.push(data_type);
        chunk.extend_from_slice(&[0; 16]);
        chunk.extend_from_slice(payload);
        chunk
    }

    fn make_test_usm(has_audio: bool) -> Vec<u8> {
        let crid = make_utf(
            "CRIUSF_DIR_STREAM",
            &[TestField::String("filename", "sample.usm")],
        );
        let video_header = make_utf(
            "VIDEO_HDRINFO",
            &[
                TestField::UInt("framerate_n", 30000),
                TestField::UInt("framerate_d", 1001),
                TestField::UInt("mpeg_codec", 9),
            ],
        );
        let audio_header = make_utf("AUDIO_HDRINFO", &[TestField::UInt("audio_codec", 2)]);
        let video_metadata = make_utf("VIDEO_SEEKINFO", &[TestField::UInt("num_skip", 0)]);

        let mut usm = make_chunk(b"CRID", &crid);
        usm.extend_from_slice(&make_chunk(b"@SFV", &video_header));
        if has_audio {
            usm.extend_from_slice(&make_chunk(b"@SFA", &audio_header));
        }
        usm.extend_from_slice(&make_chunk(b"@SFV", b"#HEADER END"));
        if has_audio {
            usm.extend_from_slice(&make_chunk(b"@SFA", b"#HEADER END"));
        }
        usm.extend_from_slice(&make_chunk(b"@SFV", &video_metadata));
        usm.extend_from_slice(&make_metadata_end_chunk());

        // A contents marker exercises the extractor's skip path without ending
        // the other stream, matching how real USMs terminate each stream.
        usm.extend_from_slice(&make_stream_chunk(b"@SFV", b"#CONTENTS END", 1));
        usm.extend_from_slice(&make_stream_chunk(b"@SFV", &[0x56; 0x300], 0));
        if has_audio {
            usm.extend_from_slice(&make_stream_chunk(b"@SFA", &[0x41; 0x180], 0));
        }
        usm.extend_from_slice(&make_stream_chunk(b"JUNK", &[0; 16], 1));
        usm
    }

    #[test]
    fn reads_complete_metadata_with_and_without_audio() {
        let metadata = read_metadata(Cursor::new(make_test_usm(true)), b"fallback.usm").unwrap();
        assert_eq!(metadata.container_filename.as_deref(), Some("sample.usm"));
        assert!(metadata.has_audio);
        assert_eq!(metadata.video_frame_rate(), Some((30000, 1001)));
        assert_eq!(metadata.sections.len(), 7);
        assert_eq!(metadata.sections[1].kind, "video_header");
        assert!(metadata.stream_offset > 0);

        let json = serde_json::to_value(&metadata).unwrap();
        assert_eq!(json["container_filename"], "sample.usm");
        assert_eq!(json["sections"][3]["data"], "#HEADER END");

        let metadata = read_metadata(Cursor::new(make_test_usm(false)), b"fallback.usm").unwrap();
        assert!(!metadata.has_audio);
        assert_eq!(metadata.sections.len(), 5);
    }

    #[test]
    fn reads_exports_and_extracts_synthetic_usm() {
        let usm = make_test_usm(true);
        let streams = extract_usm_to_memory(Cursor::new(&usm), b"fallback.usm", None, true)
            .expect("memory extraction");
        assert_eq!(streams.len(), 2);
        assert_eq!(
            (streams[0].name.as_str(), streams[0].extension.as_str()),
            ("sample", "ivf")
        );
        assert_eq!(streams[0].data, vec![0x56; 0x300]);
        assert_eq!(streams[1].extension, "adx");
        assert_eq!(streams[1].data, vec![0x41; 0x180]);

        // Exercise the in-memory masking branches as well. The synthetic data
        // is intentionally not pre-masked, so only its shape is asserted.
        let masked = extract_usm_to_memory(
            Cursor::new(&usm),
            b"fallback.usm",
            Some(0x0011_2233_4455_6677),
            true,
        )
        .unwrap();
        assert_eq!(masked[0].data.len(), 0x300);
        assert_eq!(masked[1].data.len(), 0x180);
        assert_ne!(masked[0].data, streams[0].data);
        assert_ne!(masked[1].data, streams[1].data);

        let temp = tempfile::tempdir().unwrap();
        let usm_path = temp.path().join("input.usm");
        let json_path = temp.path().join("metadata.json");
        std::fs::write(&usm_path, &usm).unwrap();

        let file_metadata = read_metadata_file(&usm_path).unwrap();
        assert_eq!(file_metadata.input_file.as_deref(), usm_path.to_str());
        export_metadata_file(&usm_path, &json_path).unwrap();
        let exported: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&json_path).unwrap()).unwrap();
        assert_eq!(exported["has_audio"], true);

        let output_dir = temp.path().join("reader-output");
        std::fs::create_dir(&output_dir).unwrap();
        let paths = extract_usm(Cursor::new(&usm), &output_dir, b"fallback.usm", None, true)
            .expect("reader extraction");
        assert_eq!(paths.len(), 2);
        assert_eq!(std::fs::read(&paths[0]).unwrap(), vec![0x56; 0x300]);

        let file_output_dir = temp.path().join("file-output");
        std::fs::create_dir(&file_output_dir).unwrap();
        let paths = extract_usm_file(&usm_path, &file_output_dir, None, false).unwrap();
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn normalizes_utf_values_and_binary_data() {
        let values = [
            UtfValue::Byte(1),
            UtfValue::SByte(-2),
            UtfValue::UShort(3),
            UtfValue::Short(-4),
            UtfValue::UInt(5),
            UtfValue::Int(-6),
            UtfValue::ULong(7),
            UtfValue::Float(8.5),
            UtfValue::String(b"text".to_vec()),
            UtfValue::Data(b"data".to_vec()),
            UtfValue::Data(vec![0xff, 0x00, 0xfe]),
        ];
        let normalized: Vec<_> = values.iter().map(normalize_metadata_value).collect();
        assert!(matches!(normalized[0], MetadataValue::Byte(1)));
        assert!(matches!(normalized[1], MetadataValue::SByte(-2)));
        assert!(matches!(normalized[2], MetadataValue::UShort(3)));
        assert!(matches!(normalized[3], MetadataValue::Short(-4)));
        assert!(matches!(normalized[4], MetadataValue::UInt(5)));
        assert!(matches!(normalized[5], MetadataValue::Int(-6)));
        assert!(matches!(normalized[6], MetadataValue::ULong(7)));
        assert!(matches!(normalized[7], MetadataValue::Float(v) if v == 8.5));
        assert!(matches!(&normalized[8], MetadataValue::String(v) if v == "text"));
        assert!(matches!(&normalized[9], MetadataValue::String(v) if v == "data"));
        assert!(matches!(&normalized[10], MetadataValue::Binary(v) if v.size == 3));

        let empty = summarize_binary(&[]);
        assert_eq!(
            (empty.size, empty.preview_hex, empty.truncated),
            (0, None, None)
        );
        let short = summarize_binary(&[1, 2, 3]);
        assert_eq!(short.preview_hex.as_deref(), Some("010203"));
        assert_eq!(short.truncated, None);
        let long = summarize_binary(&[0xab; 40]);
        assert_eq!(long.preview_hex.as_deref().map(str::len), Some(64));
        assert_eq!(long.truncated, Some(true));

        assert_eq!(decode_text_bytes(b""), Some(String::new()));
        assert_eq!(decode_text_bytes(b"hello"), Some("hello".to_string()));
        assert_eq!(
            decode_text_bytes(&[0x83, 0x65, 0x83, 0x58, 0x83, 0x67]),
            Some("テスト".to_string())
        );
        assert_eq!(decode_text_bytes(&[0xff, 0x00]), None);
        assert!(!is_mostly_text("ok\u{0001}"));
        assert!(!is_mostly_text("bad\u{fffd}"));
        assert!(is_mostly_text("line\n"));
    }

    #[test]
    fn handles_filename_fallbacks_numbers_and_errors() {
        let mut row = UtfRow::new();
        row.insert(
            "filename".to_string(),
            UtfValue::Data(b"from-data.usm".to_vec()),
        );
        assert_eq!(
            extract_container_filename(&vec![row], b"fallback.usm"),
            "from-data.usm"
        );
        assert_eq!(
            extract_container_filename(&Vec::new(), b"fallback.usm"),
            "fallback.usm"
        );
        assert_eq!(stringify_bytes(&[0xff, 0x00]), "ff00");
        assert!(stringify_utf_value(&UtfValue::UInt(1)).is_none());

        let numbers = [
            MetadataValue::Byte(1),
            MetadataValue::SByte(-2),
            MetadataValue::UShort(3),
            MetadataValue::Short(-4),
            MetadataValue::UInt(5),
            MetadataValue::Int(-6),
            MetadataValue::ULong(7),
            MetadataValue::Float(8.9),
        ];
        let converted: Vec<_> = numbers.iter().map(metadata_number_to_i32).collect();
        assert_eq!(
            converted,
            vec![
                Some(1),
                Some(-2),
                Some(3),
                Some(-4),
                Some(5),
                Some(-6),
                Some(7),
                Some(8)
            ]
        );
        assert_eq!(
            metadata_number_to_i32(&MetadataValue::String("x".into())),
            None
        );

        assert!(matches!(
            read_metadata(Cursor::new(b"NOPE".to_vec()), b"fallback"),
            Err(MetadataError::InvalidCridSignature)
        ));
        let mut invalid_utf = make_test_usm(false);
        invalid_utf[32..36].copy_from_slice(b"NOPE");
        assert!(matches!(
            read_metadata(Cursor::new(invalid_utf), b"fallback"),
            Err(MetadataError::InvalidUtfSignature)
        ));
    }

    #[test]
    fn reads_column_types_and_alignment_helpers() {
        let mut bytes = Vec::new();
        bytes.push(0xfe);
        bytes.push((-2i8) as u8);
        push_u16(&mut bytes, 0x1234);
        bytes.extend_from_slice(&(-123i16).to_be_bytes());
        push_u32(&mut bytes, 0x1234_5678);
        bytes.extend_from_slice(&(-12345i32).to_be_bytes());
        bytes.extend_from_slice(&0x1234_5678_9abc_def0u64.to_be_bytes());
        bytes.extend_from_slice(&1.5f32.to_be_bytes());
        let mut reader = Reader::new(Cursor::new(bytes));
        assert!(matches!(
            read_column_data(&mut reader, COLUMN_TYPE_1BYTE, 0, 0).unwrap(),
            UtfValue::Byte(0xfe)
        ));
        assert!(matches!(
            read_column_data(&mut reader, COLUMN_TYPE_1BYTE2, 0, 0).unwrap(),
            UtfValue::SByte(-2)
        ));
        assert!(matches!(
            read_column_data(&mut reader, COLUMN_TYPE_2BYTE, 0, 0).unwrap(),
            UtfValue::UShort(0x1234)
        ));
        assert!(matches!(
            read_column_data(&mut reader, COLUMN_TYPE_2BYTE2, 0, 0).unwrap(),
            UtfValue::Short(-123)
        ));
        assert!(matches!(
            read_column_data(&mut reader, COLUMN_TYPE_4BYTE, 0, 0).unwrap(),
            UtfValue::UInt(0x1234_5678)
        ));
        assert!(matches!(
            read_column_data(&mut reader, COLUMN_TYPE_4BYTE2, 0, 0).unwrap(),
            UtfValue::Int(-12345)
        ));
        assert!(matches!(
            read_column_data(&mut reader, COLUMN_TYPE_8BYTE, 0, 0).unwrap(),
            UtfValue::ULong(0x1234_5678_9abc_def0)
        ));
        assert!(
            matches!(read_column_data(&mut reader, COLUMN_TYPE_FLOAT, 0, 0).unwrap(), UtfValue::Float(v) if v == 1.5)
        );
        assert!(matches!(
            read_column_data(&mut reader, 0x0f, 0, 0),
            Err(MetadataError::Usm(UsmError::UnknownColumnType(0x0f)))
        ));

        let mut cstring = Reader::new(Cursor::new(b"abc\0tail".to_vec()));
        assert_eq!(read_cstring(&mut cstring).unwrap(), b"abc");
        let mut aligned = Reader::new(Cursor::new(vec![0; 16]));
        aligned.seek(SeekFrom::Start(3)).unwrap();
        assert_eq!(align_stream(&mut aligned, 4).unwrap(), 4);
        assert_eq!(align_stream(&mut aligned, 4).unwrap(), 4);
        assert_eq!(
            read_signature_at(&mut Reader::new(Cursor::new(b"xxxx@SFV".to_vec())), 4).unwrap(),
            "@SFV"
        );
        assert!(seek_and_check_signature(
            &mut Reader::new(Cursor::new(b"NOPE".to_vec())),
            0,
            "@SFV"
        )
        .is_err());
    }
}
