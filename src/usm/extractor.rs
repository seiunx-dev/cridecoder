//! USM video/audio extraction

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::reader::Reader;
use encoding_rs::SHIFT_JIS;
use thiserror::Error;

/// Column storage masks
const COLUMN_STORAGE_MASK: u8 = 0xF0;
const _COLUMN_STORAGE_PERROW: u8 = 0x50;
const COLUMN_STORAGE_CONSTANT: u8 = 0x30;
const COLUMN_STORAGE_CONSTANT2: u8 = 0x70;
const _COLUMN_STORAGE_ZERO: u8 = 0x10;

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
const MASK_LEN: usize = 0x20;

type VideoMask = ([u8; MASK_LEN], [u8; MASK_LEN]);
type AudioMask = [u8; MASK_LEN];

/// (filename, has_audio, mpeg_codec, audio_codec) parsed from the USM CRID header.
type UsmHeaderInfo = (Vec<u8>, bool, Option<u32>, Option<u32>);

/// USM extraction errors
#[derive(Debug, Error)]
pub enum UsmError {
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
    #[error("unknown column type: {0}")]
    UnknownColumnType(u8),
}

/// A value from a UTF table row
#[derive(Debug, Clone)]
pub enum UtfValue {
    Byte(u8),
    SByte(i8),
    UShort(u16),
    Short(i16),
    UInt(u32),
    Int(i32),
    ULong(u64),
    Float(f32),
    String(Vec<u8>),
    Data(Vec<u8>),
}

/// A row in a UTF table
pub type UtfRow = std::collections::HashMap<String, UtfValue>;

/// A UTF table
pub type UtfTable = Vec<UtfRow>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedUsmStream {
    pub name: String,
    pub extension: String,
    pub data: Vec<u8>,
}

/// Read the first row's named integer column as u32 (codec fields), if present.
fn utf_first_u32(table: &UtfTable, key: &str) -> Option<u32> {
    Some(match table.first()?.get(key)? {
        UtfValue::Byte(v) => *v as u32,
        UtfValue::SByte(v) => *v as u32,
        UtfValue::UShort(v) => *v as u32,
        UtfValue::Short(v) => *v as u32,
        UtfValue::UInt(v) => *v,
        UtfValue::Int(v) => *v as u32,
        UtfValue::ULong(v) => *v as u32,
        _ => return None,
    })
}

/// @SFA AUDIO_HDRINFO.audio_codec: 2 = ADX, 4 = HCA (PyCriCodecs usm.py:168).
/// Defaults to adx when the column is absent (e.g. minimal builder-made USMs).
fn audio_ext(audio_codec: Option<u32>) -> &'static str {
    match audio_codec {
        Some(4) => "hca",
        _ => "adx",
    }
}

/// @SFV VIDEO_HDRINFO.mpeg_codec: 9 = VP9 (pjsk) -> ivf; default m2v (MPEG2).
fn video_ext(mpeg_codec: Option<u32>) -> &'static str {
    match mpeg_codec {
        Some(9) => "ivf",
        _ => "m2v",
    }
}

/// Read column data from a UTF table
fn read_column_data<R: Read + Seek>(
    reader: &mut Reader<R>,
    column_type: u8,
    string_table_offset: i64,
    data_offset: i64,
) -> Result<UtfValue, UsmError> {
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
        _ => Err(UsmError::UnknownColumnType(column_type)),
    }
}

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

/// Field info for UTF table parsing
struct FieldInfo {
    name: String,
    column_type: u8,
    constant: Option<UtfValue>,
}

/// Parse a UTF table from the reader
fn get_utf_table<R: Read + Seek>(reader: &mut Reader<R>) -> Result<UtfTable, UsmError> {
    let sig = reader.read_bytes(4)?;
    if &sig != b"@UTF" {
        return Err(UsmError::InvalidUtfSignature);
    }

    let table_size = reader.read_u32()?;
    let _version = reader.read_u16()?;
    let row_offset = reader.read_u16()?;
    let string_table_offset = reader.read_u32()?;
    let data_offset = reader.read_u32()?;
    let _table_name_offset = reader.read_u32()?;
    let number_of_fields = reader.read_u16()?;
    let _row_size = reader.read_u16()?;
    let number_of_rows = reader.read_u32()?;

    let table_data = reader.read_bytes((table_size - 24) as usize)?;
    let mut utf_reader = Reader::new(io::Cursor::new(table_data));

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

    Ok(rows)
}

/// Generate mask from key
fn get_mask(key: u64) -> (VideoMask, AudioMask) {
    let key1 = (key & 0xFFFFFFFF) as u32;
    let key2 = ((key >> 32) & 0xFFFFFFFF) as u32;

    let mut t = [0u8; MASK_LEN];
    t[0x00] = (key1 & 0xFF) as u8;
    t[0x01] = ((key1 >> 8) & 0xFF) as u8;
    t[0x02] = ((key1 >> 16) & 0xFF) as u8;
    t[0x03] = (((key1 >> 24) & 0xFF) as u8).wrapping_sub(0x34);
    t[0x04] = ((key2 & 0xF) as u8).wrapping_add(0xF9);
    t[0x05] = ((key2 >> 8) & 0xFF) as u8 ^ 0x13;
    t[0x06] = (((key2 >> 16) & 0xFF) as u8).wrapping_add(0x61);
    t[0x07] = t[0x00] ^ 0xFF;
    t[0x08] = (t[0x02] as u16 + t[0x01] as u16) as u8;
    t[0x09] = (t[0x01] as i16 - t[0x07] as i16) as u8;
    t[0x0A] = t[0x02] ^ 0xFF;
    t[0x0B] = t[0x01] ^ 0xFF;
    t[0x0C] = (t[0x0B] as u16 + t[0x09] as u16) as u8;
    t[0x0D] = (t[0x08] as i16 - t[0x03] as i16) as u8;
    t[0x0E] = t[0x0D] ^ 0xFF;
    t[0x0F] = (t[0x0A] as i16 - t[0x0B] as i16) as u8;
    t[0x10] = (t[0x08] as i16 - t[0x0F] as i16) as u8;
    t[0x11] = t[0x10] ^ t[0x07];
    t[0x12] = t[0x0F] ^ 0xFF;
    t[0x13] = t[0x03] ^ 0x10;
    t[0x14] = (t[0x04] as i16 - 0x32) as u8;
    t[0x15] = (t[0x05] as u16 + 0xED) as u8;
    t[0x16] = t[0x06] ^ 0xF3;
    t[0x17] = (t[0x13] as i16 - t[0x0F] as i16) as u8;
    t[0x18] = (t[0x15] as u16 + t[0x07] as u16) as u8;
    t[0x19] = (0x21i16 - t[0x13] as i16) as u8;
    t[0x1A] = t[0x14] ^ t[0x17];
    t[0x1B] = (t[0x16] as u16 + t[0x16] as u16) as u8;
    t[0x1C] = (t[0x17] as u16 + 0x44) as u8;
    t[0x1D] = (t[0x03] as u16 + t[0x04] as u16) as u8;
    t[0x1E] = (t[0x05] as i16 - t[0x16] as i16) as u8;
    t[0x1F] = t[0x1D] ^ t[0x13];

    let t2 = b"URUC";
    let mut vmask1 = [0u8; MASK_LEN];
    let mut vmask2 = [0u8; MASK_LEN];
    let mut amask = [0u8; MASK_LEN];

    for (i, &ti) in t.iter().enumerate() {
        vmask1[i] = ti;
        vmask2[i] = ti ^ 0xFF;
        if i & 1 != 0 {
            amask[i] = t2[(i >> 1) & 3];
        } else {
            amask[i] = ti ^ 0xFF;
        }
    }

    ((vmask1, vmask2), amask)
}

/// De-mask video content **in place**.
///
/// The bulk (second) pass is written as a 32-byte (mask-width) chunked XOR so
/// LLVM auto-vectorizes it to SSE2/NEON/AVX with no platform-specific
/// intrinsics. The per-lane mask recurrence is preserved exactly: the 32 lanes
/// are independent, so each 32-byte row does `row ^= mask; mask = row ^ vmask1`
/// as two vector ops with `mask` carried in a register across rows. The first
/// pass (256 B, negligible) stays scalar to avoid an aliasing split. Output is
/// bit-identical to the original scalar version (locked by `mask_golden`).
fn mask_video(buf: &mut [u8], vmask: &VideoMask) {
    let len = buf.len();
    if len.saturating_sub(0x40) < 0x200 {
        return;
    }
    let vm1 = vmask.1;
    let mut mask = vmask.1;

    // Second pass: original i in 0x100..size -> buf[0x140..len]. 0x100 is
    // 32-aligned, so row position j maps to mask lane j.
    {
        let (chunks, remainder) = buf[0x140..len].as_chunks_mut::<MASK_LEN>();
        for row in chunks {
            for j in 0..MASK_LEN {
                row[j] ^= mask[j];
                mask[j] = row[j] ^ vm1[j];
            }
        }
        for (j, b) in remainder.iter_mut().enumerate() {
            *b ^= mask[j];
            mask[j] = *b ^ vm1[j];
        }
    }

    // First pass: i in 0..0x100 reads the now-decoded buf[0x140..0x240].
    let mut mask = vmask.0;
    for i in 0..0x100 {
        let v = buf[0x140 + i];
        let l = i & 0x1F;
        mask[l] ^= v;
        buf[0x40 + i] ^= mask[l];
    }
}

/// De-mask audio content **in place** (simple repeating 32-byte XOR). Chunked
/// to the mask width so LLVM auto-vectorizes. Bit-identical to the original.
fn mask_audio(buf: &mut [u8], amask: &AudioMask) {
    let Some(region) = buf.get_mut(0x140..) else {
        return;
    };
    let (chunks, remainder) = region.as_chunks_mut::<MASK_LEN>();
    for row in chunks {
        for j in 0..MASK_LEN {
            row[j] ^= amask[j];
        }
    }
    for (j, b) in remainder.iter_mut().enumerate() {
        *b ^= amask[j];
    }
}

/// Decode Shift-JIS bytes to String
fn decode_shift_jis(data: &[u8]) -> String {
    let (decoded, _, _) = SHIFT_JIS.decode(data);
    decoded.to_string()
}

/// Extract USM from a reader
pub fn extract_usm<R: Read + Seek>(
    usm: R,
    target_dir: &Path,
    fallback_name: &[u8],
    key: Option<u64>,
    export_audio: bool,
) -> Result<Vec<PathBuf>, UsmError> {
    let mut reader = Reader::new(usm);

    let (vmask, amask) = key.map(get_mask).unzip();

    let (filename, has_audio, mpeg_codec, audio_codec) =
        parse_usm_header(&mut reader, fallback_name)?;
    let decoded_filename = decode_shift_jis(&filename);

    let (video_file, audio_file, output_files) = create_output_files(
        target_dir,
        &decoded_filename,
        has_audio,
        export_audio,
        mpeg_codec,
        audio_codec,
    )?;

    // The USM audio XOR mask applies only to ADX (codec 2); HCA has its own
    // internal cipher, so masking it would corrupt the stream (PyCriCodecs usm.py:273).
    let amask = if audio_codec == Some(2) { amask } else { None };

    // Buffer the outputs: chunk payloads are ~32 KiB and io::copy uses an
    // 8 KiB buffer, so unbuffered files pay several write syscalls per chunk.
    let mut video_out = io::BufWriter::with_capacity(1 << 20, video_file);
    let mut audio_out = audio_file.map(|f| io::BufWriter::with_capacity(1 << 20, f));

    extract_usm_chunks(
        &mut reader,
        &mut video_out,
        audio_out.as_mut(),
        vmask.as_ref(),
        amask.as_ref(),
    )?;

    video_out.flush()?;
    if let Some(audio_out) = audio_out.as_mut() {
        audio_out.flush()?;
    }

    Ok(output_files)
}

/// Extract USM video/audio streams from a reader into memory.
pub fn extract_usm_to_memory<R: Read + Seek>(
    usm: R,
    fallback_name: &[u8],
    key: Option<u64>,
    export_audio: bool,
) -> Result<Vec<ExtractedUsmStream>, UsmError> {
    let mut reader = Reader::new(usm);
    let (vmask, amask) = key.map(get_mask).unzip();
    let (filename, has_audio, mpeg_codec, audio_codec) =
        parse_usm_header(&mut reader, fallback_name)?;
    let decoded_filename = decode_shift_jis(&filename);
    let base_name = Path::new(&decoded_filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&decoded_filename)
        .to_string();

    // Audio mask only applies to ADX (codec 2); HCA carries its own cipher.
    let amask = if audio_codec == Some(2) { amask } else { None };

    extract_usm_chunks_to_memory(
        &mut reader,
        base_name,
        has_audio && export_audio,
        vmask.as_ref(),
        amask.as_ref(),
        mpeg_codec,
        audio_codec,
    )
}

/// Parse USM header
fn parse_usm_header<R: Read + Seek>(
    reader: &mut Reader<R>,
    fallback_name: &[u8],
) -> Result<UsmHeaderInfo, UsmError> {
    let sig = reader.read_bytes(4)?;
    if &sig != b"CRID" {
        return Err(UsmError::InvalidCridSignature);
    }

    let block_size = reader.read_u32()?;
    reader.seek(SeekFrom::Start(0x20))?;
    let entry_table = get_utf_table(reader)?;

    let filename = extract_filename(&entry_table, fallback_name);
    let offset = 8 + block_size as i64;

    let (has_audio, mpeg_codec, audio_codec, offset) = parse_usm_header_chunks(reader, offset)?;
    skip_metadata_section(reader, offset)?;

    Ok((filename, has_audio, mpeg_codec, audio_codec))
}

/// Extract filename from entry table
fn extract_filename(entry_table: &UtfTable, fallback_name: &[u8]) -> Vec<u8> {
    if let Some(row) = entry_table.last() {
        if let Some(UtfValue::String(filename)) = row.get("filename") {
            return filename.clone();
        }
        if let Some(UtfValue::Data(filename)) = row.get("filename") {
            return filename.clone();
        }
    }
    fallback_name.to_vec()
}

/// Parse USM header chunks
fn parse_usm_header_chunks<R: Read + Seek>(
    reader: &mut Reader<R>,
    mut offset: i64,
) -> Result<(bool, Option<u32>, Option<u32>, i64), UsmError> {
    // First @SFV chunk
    seek_and_check_signature(reader, offset, "@SFV")?;
    let block_size = reader.read_u32()?;
    reader.seek(SeekFrom::Start((offset + 0x20) as u64))?;
    let sfv_table = get_utf_table(reader)?;
    let mpeg_codec = utf_first_u32(&sfv_table, "mpeg_codec");
    let mut audio_codec: Option<u32> = None;
    offset += 8 + block_size as i64;

    // Check for optional @SFA chunk
    reader.seek(SeekFrom::Start(offset as u64))?;
    let next_sig = reader.read_bytes(4)?;
    let mut has_audio = false;

    let mut next_sig = next_sig;
    if &next_sig == b"@SFA" {
        let block_size = reader.read_u32()?;
        reader.seek(SeekFrom::Start((offset + 0x20) as u64))?;
        let sfa_table = get_utf_table(reader)?;
        audio_codec = utf_first_u32(&sfa_table, "audio_codec");
        offset += 8 + block_size as i64;
        has_audio = true;
        reader.seek(SeekFrom::Start(offset as u64))?;
        next_sig = reader.read_bytes(4)?;
    }

    // Second @SFV with HEADER END
    if &next_sig != b"@SFV" {
        return Err(UsmError::ExpectedSignature("@SFV".to_string()));
    }
    let block_size = reader.read_u32()?;
    reader.seek(SeekFrom::Start((offset + 0x20) as u64))?;
    let header_end = reader.read_bytes(11)?;
    if &header_end != b"#HEADER END" {
        return Err(UsmError::ExpectedMarker("#HEADER END".to_string()));
    }
    offset += 8 + block_size as i64;

    // Optional @SFA with HEADER END
    if has_audio {
        seek_and_check_signature(reader, offset, "@SFA")?;
        let block_size = reader.read_u32()?;
        reader.seek(SeekFrom::Start((offset + 0x20) as u64))?;
        let header_end = reader.read_bytes(11)?;
        if &header_end != b"#HEADER END" {
            return Err(UsmError::ExpectedMarker("#HEADER END".to_string()));
        }
        offset += 8 + block_size as i64;
    }

    Ok((has_audio, mpeg_codec, audio_codec, offset))
}

/// Seek to offset and check signature
fn seek_and_check_signature<R: Read + Seek>(
    reader: &mut Reader<R>,
    offset: i64,
    expected: &str,
) -> Result<(), UsmError> {
    reader.seek(SeekFrom::Start(offset as u64))?;
    let sig = reader.read_bytes(4)?;
    if sig != expected.as_bytes() {
        return Err(UsmError::ExpectedSignature(expected.to_string()));
    }
    Ok(())
}

/// Skip metadata section
fn skip_metadata_section<R: Read + Seek>(
    reader: &mut Reader<R>,
    mut offset: i64,
) -> Result<(), UsmError> {
    // First metadata @SFV
    seek_and_check_signature(reader, offset, "@SFV")?;
    let block_size = reader.read_u32()?;
    reader.seek(SeekFrom::Start((offset + 0x20) as u64))?;
    let _ = get_utf_table(reader)?;
    offset += 8 + block_size as i64;

    // Second metadata @SFV with METADATA END
    seek_and_check_signature(reader, offset, "@SFV")?;
    reader.seek(SeekFrom::Current(28))?;
    let metadata_end = reader.read_bytes(13)?;
    if &metadata_end != b"#METADATA END" {
        return Err(UsmError::ExpectedMarker("#METADATA END".to_string()));
    }
    align_stream(reader, 4)?;
    reader.seek(SeekFrom::Current(16))?;

    Ok(())
}

/// Create output files
fn create_output_files(
    target_dir: &Path,
    decoded_filename: &str,
    has_audio: bool,
    export_audio: bool,
    mpeg_codec: Option<u32>,
    audio_codec: Option<u32>,
) -> Result<(File, Option<File>, Vec<PathBuf>), UsmError> {
    let base_name = Path::new(decoded_filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(decoded_filename);

    let video_path = target_dir.join(format!("{}.{}", base_name, video_ext(mpeg_codec)));
    let video_file = File::create(&video_path)?;

    let mut output_files = vec![video_path];
    let audio_file = if has_audio && export_audio {
        let audio_path = target_dir.join(format!("{}.{}", base_name, audio_ext(audio_codec)));
        let file = File::create(&audio_path)?;
        output_files.push(audio_path);
        Some(file)
    } else {
        None
    };

    Ok((video_file, audio_file, output_files))
}

/// Extract USM chunks
fn extract_usm_chunks<R: Read + Seek, W: Write>(
    reader: &mut Reader<R>,
    video_file: &mut W,
    mut audio_file: Option<&mut W>,
    vmask: Option<&VideoMask>,
    amask: Option<&AudioMask>,
) -> Result<(), UsmError> {
    while let Ok(next_sig) = reader.read_bytes(4) {
        let block_size = reader.read_u32()?;
        let current_pos = reader.stream_position()?;
        let next_offset = current_pos + block_size as u64;

        let chunk_header_size = reader.read_u16()?;
        let chunk_footer_size = reader.read_u16()?;
        let _ = reader.read_bytes(3)?;
        let data_type_byte = reader.read_i8()?;
        let data_type = (data_type_byte & 0b11) as u8;
        reader.seek(SeekFrom::Current(16))?;

        let contents_end = reader.read_bytes(13)?;
        if &contents_end == b"#CONTENTS END" {
            // Each stream emits its own #CONTENTS END; skip it and keep reading to
            // EOF. Breaking here truncates the longer stream (PyCriCodecs usm.py).
            reader.seek(SeekFrom::Start(next_offset))?;
            continue;
        }

        reader.seek(SeekFrom::Current(-13))?;
        let read_data_len =
            block_size as usize - chunk_header_size as usize - chunk_footer_size as usize;

        process_chunk(
            reader,
            &next_sig,
            read_data_len,
            data_type,
            video_file,
            &mut audio_file,
            vmask,
            amask,
        )?;

        reader.seek(SeekFrom::Start(next_offset))?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn extract_usm_chunks_to_memory<R: Read + Seek>(
    reader: &mut Reader<R>,
    base_name: String,
    export_audio: bool,
    vmask: Option<&VideoMask>,
    amask: Option<&AudioMask>,
    mpeg_codec: Option<u32>,
    audio_codec: Option<u32>,
) -> Result<Vec<ExtractedUsmStream>, UsmError> {
    // Pre-reserve using the remaining stream length: payload is the file
    // minus headers/footers, so this slightly over-reserves but avoids every
    // growth realloc (the payload is tens of MB).
    let here = reader.stream_position()?;
    let total = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(here))?;
    let remaining = (total - here) as usize;

    let mut video = Vec::with_capacity(remaining);
    let mut audio = if export_audio { Some(Vec::new()) } else { None };

    while let Ok(next_sig) = reader.read_bytes(4) {
        let block_size = reader.read_u32()?;
        let current_pos = reader.stream_position()?;
        let next_offset = current_pos + block_size as u64;

        let chunk_header_size = reader.read_u16()?;
        let chunk_footer_size = reader.read_u16()?;
        let _ = reader.read_bytes(3)?;
        let data_type_byte = reader.read_i8()?;
        let data_type = (data_type_byte & 0b11) as u8;
        reader.seek(SeekFrom::Current(16))?;

        let contents_end = reader.read_bytes(13)?;
        if &contents_end == b"#CONTENTS END" {
            // Each stream emits its own #CONTENTS END; skip it and keep reading to
            // EOF. Breaking here truncates the longer stream (PyCriCodecs usm.py).
            reader.seek(SeekFrom::Start(next_offset))?;
            continue;
        }

        reader.seek(SeekFrom::Current(-13))?;
        let read_data_len =
            block_size as usize - chunk_header_size as usize - chunk_footer_size as usize;

        if next_sig == b"@SFV" {
            read_usm_chunk_into(reader, read_data_len, data_type, vmask, None, &mut video)?;
        } else if next_sig == b"@SFA" {
            if let Some(audio) = audio.as_mut() {
                read_usm_chunk_into(reader, read_data_len, data_type, None, amask, audio)?;
            }
        }

        reader.seek(SeekFrom::Start(next_offset))?;
    }

    let mut streams = vec![ExtractedUsmStream {
        name: base_name.clone(),
        extension: video_ext(mpeg_codec).to_string(),
        data: video,
    }];
    if let Some(audio) = audio {
        streams.push(ExtractedUsmStream {
            name: base_name,
            extension: audio_ext(audio_codec).to_string(),
            data: audio,
        });
    }

    Ok(streams)
}

/// Read one chunk payload straight into the tail of `out` (single copy from
/// the source, no intermediate buffer), de-masking in place when needed.
fn read_usm_chunk_into<R: Read + Seek>(
    reader: &mut Reader<R>,
    read_data_len: usize,
    data_type: u8,
    vmask: Option<&VideoMask>,
    amask: Option<&AudioMask>,
    out: &mut Vec<u8>,
) -> Result<(), UsmError> {
    let start = out.len();
    reader.read_into_vec(read_data_len, out)?;
    if data_type != 0 {
        return Ok(());
    }
    if let Some(vmask) = vmask {
        mask_video(&mut out[start..], vmask);
    } else if let Some(amask) = amask {
        mask_audio(&mut out[start..], amask);
    }
    Ok(())
}

/// Process a chunk
#[allow(clippy::too_many_arguments)]
fn process_chunk<R: Read + Seek, W: Write>(
    reader: &mut Reader<R>,
    sig: &[u8],
    read_data_len: usize,
    data_type: u8,
    video_file: &mut W,
    audio_file: &mut Option<&mut W>,
    vmask: Option<&VideoMask>,
    amask: Option<&AudioMask>,
) -> Result<(), UsmError> {
    if sig == b"@SFV" {
        if data_type == 0 {
            if let Some(vmask) = vmask {
                let mut content = reader.read_bytes(read_data_len)?;
                mask_video(&mut content, vmask);
                video_file.write_all(&content)?;
            } else {
                reader.copy_to_writer(read_data_len as u64, video_file)?;
            }
        } else {
            reader.copy_to_writer(read_data_len as u64, video_file)?;
        }
    } else if sig == b"@SFA" {
        if let Some(audio_file) = audio_file {
            if data_type == 0 {
                if let Some(amask) = amask {
                    let mut content = reader.read_bytes(read_data_len)?;
                    mask_audio(&mut content, amask);
                    audio_file.write_all(&content)?;
                } else {
                    reader.copy_to_writer(read_data_len as u64, audio_file)?;
                }
            } else {
                reader.copy_to_writer(read_data_len as u64, audio_file)?;
            }
        }
    }

    Ok(())
}

/// Extract USM from a file path
pub fn extract_usm_file(
    usm_path: &Path,
    target_dir: &Path,
    key: Option<u64>,
    export_audio: bool,
) -> Result<Vec<PathBuf>, UsmError> {
    let file = File::open(usm_path)?;
    let fallback_name = usm_path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.as_bytes().to_vec())
        .unwrap_or_default();
    extract_usm(file, target_dir, &fallback_name, key, export_audio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_extension_mapping() {
        // Audio: 2=ADX, 4=HCA (PyCriCodecs usm.py:168); default adx.
        assert_eq!(audio_ext(Some(2)), "adx");
        assert_eq!(audio_ext(Some(4)), "hca");
        assert_eq!(audio_ext(None), "adx");
        // Video: 9=VP9 -> ivf (pjsk); default m2v.
        assert_eq!(video_ext(Some(9)), "ivf");
        assert_eq!(video_ext(Some(1)), "m2v");
        assert_eq!(video_ext(None), "m2v");
    }

    #[test]
    fn test_utf_first_u32_reads_codec_column() {
        let mut row = std::collections::HashMap::new();
        row.insert("audio_codec".to_string(), UtfValue::Byte(4));
        row.insert("mpeg_codec".to_string(), UtfValue::UInt(9));
        let table: UtfTable = vec![row];
        assert_eq!(utf_first_u32(&table, "audio_codec"), Some(4));
        assert_eq!(utf_first_u32(&table, "mpeg_codec"), Some(9));
        assert_eq!(utf_first_u32(&table, "missing"), None);

        // Non-integer columns and empty tables yield None.
        let mut row2 = std::collections::HashMap::new();
        row2.insert("name".to_string(), UtfValue::String(b"x".to_vec()));
        assert_eq!(utf_first_u32(&vec![row2], "name"), None);
        assert_eq!(utf_first_u32(&Vec::new(), "audio_codec"), None);
    }

    // --- mask de-XOR microbench + golden regression guard ---

    // Arbitrary synthetic mask key — only needs to exercise the de-mask paths;
    // not tied to any real content.
    const TEST_KEY: u64 = 0x0011_2233_4455_6677;

    fn lcg_fill(n: usize) -> Vec<u8> {
        // deterministic pseudo-random payload (content-independent XOR, but
        // realistic byte distribution; reproducible across runs)
        let mut v = Vec::with_capacity(n);
        let mut s: u64 = 0x243f6a8885a308d3;
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v.push((s >> 33) as u8);
        }
        v
    }

    fn fnv(data: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in data {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
        h
    }

    // Original scalar algorithm, kept verbatim as the reference oracle.
    fn ref_video(content: &[u8], vmask: &VideoMask) -> Vec<u8> {
        let mut result = content.to_vec();
        let size = result.len().saturating_sub(0x40);
        let base = 0x40;
        if size >= 0x200 {
            let mut mask = vmask.1;
            for i in 0x100..size {
                result[base + i] ^= mask[i & 0x1F];
                mask[i & 0x1F] = result[base + i] ^ vmask.1[i & 0x1F];
            }
            mask.copy_from_slice(&vmask.0);
            for i in 0..0x100 {
                mask[i & 0x1F] ^= result[0x100 + base + i];
                result[base + i] ^= mask[i & 0x1F];
            }
        }
        result
    }
    fn ref_audio(content: &[u8], amask: &AudioMask) -> Vec<u8> {
        let mut result = content.to_vec();
        let size = result.len().saturating_sub(0x140);
        let base = 0x140;
        for i in 0..size {
            result[base + i] ^= amask[i & 0x1F];
        }
        result
    }

    #[test]
    fn mask_matches_reference() {
        // Differential test: the vectorized in-place fns must be byte-identical
        // to the original scalar oracle across boundary sizes (0x40/0x140/0x200
        // thresholds, 32-byte remainders, tiny/empty buffers).
        let (vmask, amask) = get_mask(TEST_KEY);
        let sizes = [
            0,
            1,
            0x3f,
            0x40,
            0x41,
            0x13f,
            0x140,
            0x141,
            0x1ff,
            0x200,
            0x23e,
            0x23f,
            0x240,
            0x241,
            0x25f,
            0x260,
            0x261,
            0x300,
            1000,
            4096,
            4097,
            0x40 + 0x200,
            0x40 + 0x201,
            0x40 + 0x21f,
            0x40 + 0x220,
            100_000,
            100_003,
        ];
        for &n in &sizes {
            let content = lcg_fill(n);
            let mut v = content.clone();
            mask_video(&mut v, &vmask);
            assert_eq!(
                v,
                ref_video(&content, &vmask),
                "mask_video mismatch at size {n}"
            );
            let mut a = content.clone();
            mask_audio(&mut a, &amask);
            assert_eq!(
                a,
                ref_audio(&content, &amask),
                "mask_audio mismatch at size {n}"
            );
        }
    }

    #[test]
    fn mask_golden() {
        // Locks the exact byte output of mask_video/mask_audio so the SIMD/
        // auto-vectorization refactor stays bit-identical. Uses a synthetic key (TEST_KEY).
        let (vmask, amask) = get_mask(TEST_KEY);
        let content = lcg_fill(0x40 + 1_000_000); // > 0x200 so both passes run
        let mut v = content.clone();
        mask_video(&mut v, &vmask);
        let mut a = content.clone();
        mask_audio(&mut a, &amask);
        let (cv, ca) = (fnv(&v), fnv(&a));
        eprintln!("GOLDEN video={:#018x} audio={:#018x}", cv, ca);
        assert_eq!(cv, 0x01cb2bb02df7f48c, "mask_video output changed");
        assert_eq!(ca, 0x3b4a709ace2862ed, "mask_audio output changed");
    }

    #[test]
    fn bench_mask() {
        if std::env::var("USM_BENCH").is_err() {
            return;
        }
        let (vmask, amask) = get_mask(TEST_KEY);
        let n = std::env::var("USM_BENCH_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(48);
        let iters = std::env::var("USM_BENCH_ITERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(50);
        let content = lcg_fill(0x40 + n * 1024 * 1024);
        let mb = content.len() as f64 / 1e6;
        // In-place on owned buffers (no per-iter copy): this is exactly what the
        // caller now pays, since read_bytes already owns the buffer. Running the
        // same buffer repeatedly is fine for timing — the XOR work is constant.
        let mut vbuf = content.clone();
        let mut abuf = content.clone();

        // warm
        mask_video(&mut vbuf, &vmask);
        mask_audio(&mut abuf, &amask);
        std::hint::black_box((&vbuf, &abuf));

        let t = std::time::Instant::now();
        for _ in 0..iters {
            mask_video(&mut vbuf, &vmask);
            std::hint::black_box(&vbuf);
        }
        let ev = t.elapsed();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            mask_audio(&mut abuf, &amask);
            std::hint::black_box(&abuf);
        }
        let ea = t.elapsed();
        eprintln!(
            "mask_video: {:.3} ms/call, {:.0} MB/s  ({} MB x {} iters)",
            ev.as_secs_f64() * 1000.0 / iters as f64,
            mb * iters as f64 / ev.as_secs_f64(),
            n,
            iters
        );
        eprintln!(
            "mask_audio: {:.3} ms/call, {:.0} MB/s  ({} MB x {} iters)",
            ea.as_secs_f64() * 1000.0 / iters as f64,
            mb * iters as f64 / ea.as_secs_f64(),
            n,
            iters
        );
    }
}
