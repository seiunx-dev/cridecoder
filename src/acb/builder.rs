//! ACB/AWB encoder - builds CRI audio containers from audio files

use crate::acb::consts::{
    COLUMN_FLAG_DEFAULT, COLUMN_FLAG_NAME, COLUMN_FLAG_ROW, COLUMN_TYPE_1BYTE, COLUMN_TYPE_1BYTE2,
    COLUMN_TYPE_2BYTE, COLUMN_TYPE_2BYTE2, COLUMN_TYPE_4BYTE, COLUMN_TYPE_4BYTE2,
    COLUMN_TYPE_8BYTE, COLUMN_TYPE_DATA, COLUMN_TYPE_FLOAT, COLUMN_TYPE_STRING,
    WAVEFORM_ENCODE_TYPE_ADX, WAVEFORM_ENCODE_TYPE_HCA,
};
use crate::acb::utf::Value;
use encoding_rs::SHIFT_JIS;
use std::collections::{HashMap, HashSet};
use std::io::{self, Seek, Write};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BuilderError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("No audio tracks provided")]
    NoTracks,
    #[error("Track name too long: {0}")]
    NameTooLong(String),
    #[error("Invalid audio data")]
    InvalidAudioData,
    #[error("Unsupported audio format")]
    UnsupportedFormat,
    #[error("Too many tracks for ACB builder: {0}")]
    TooManyTracks(usize),
    #[error("Cue ID exceeds WaveformTable 16-bit limit: {0}")]
    CueIdTooLarge(u32),
    #[error("Duplicate cue ID: {0}")]
    DuplicateCueId(u32),
    #[error("music ACB builder requires exactly one track, got {0}")]
    MusicAcbRequiresSingleTrack(usize),
    #[error("{field} must be {expected} bytes, got {actual}")]
    InvalidFixedDataLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

/// Audio track input for ACB building
#[derive(Debug, Clone)]
pub struct TrackInput {
    pub name: String,
    pub cue_id: u32,
    pub data: Vec<u8>,
    pub encode_type: i32,
}

impl TrackInput {
    pub fn new(name: impl Into<String>, cue_id: u32, data: Vec<u8>) -> Self {
        let encode_type = Self::detect_encode_type(&data);
        Self {
            name: name.into(),
            cue_id,
            data,
            encode_type,
        }
    }

    fn detect_encode_type(data: &[u8]) -> i32 {
        if data.len() < 4 {
            return WAVEFORM_ENCODE_TYPE_HCA;
        }
        match &data[0..4] {
            [0x80, 0x00, ..] | [0x00, 0x00, 0x00, 0x80] => WAVEFORM_ENCODE_TYPE_ADX,
            b"HCA\x00" | [0xc8, 0xc3, 0xc1, 0x80] => WAVEFORM_ENCODE_TYPE_HCA, // HCA or masked HCA
            _ if data.len() > 6 && data[0] == 0xC8 => WAVEFORM_ENCODE_TYPE_HCA, // masked HCA header
            _ => WAVEFORM_ENCODE_TYPE_HCA,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct HcaTrackInfo {
    channels: u8,
    sampling_rate: u16,
    num_samples: u32,
    length_ms: u32,
}

impl HcaTrackInfo {
    fn from_hca_with_reference(
        data: &[u8],
        reference_num_samples: u32,
        reference_length_ms: u32,
    ) -> Self {
        let Some(fmt_offset) = data.windows(4).position(|window| window == b"fmt\0") else {
            return Self::with_reference(reference_num_samples, reference_length_ms);
        };

        if fmt_offset + 12 > data.len() {
            return Self::with_reference(reference_num_samples, reference_length_ms);
        }

        let channels = data[fmt_offset + 4].max(1);
        let sampling_rate = u32::from_be_bytes([
            0,
            data[fmt_offset + 5],
            data[fmt_offset + 6],
            data[fmt_offset + 7],
        ]);
        let frame_count = u32::from_be_bytes([
            data[fmt_offset + 8],
            data[fmt_offset + 9],
            data[fmt_offset + 10],
            data[fmt_offset + 11],
        ]);

        if sampling_rate == 0 || frame_count == 0 {
            return Self::with_reference(reference_num_samples, reference_length_ms);
        }

        let num_samples = frame_count.saturating_mul(1024);
        let length_ms = ((num_samples as u64 * 1000) / sampling_rate as u64) as u32;
        if length_ms >= reference_length_ms {
            return Self::with_reference(reference_num_samples, reference_length_ms);
        }

        Self {
            channels,
            sampling_rate: sampling_rate.min(u16::MAX as u32) as u16,
            num_samples: if num_samples == 0 {
                reference_num_samples
            } else {
                num_samples
            },
            length_ms: if length_ms == 0 {
                reference_length_ms
            } else {
                length_ms
            },
        }
    }

    fn with_reference(num_samples: u32, length_ms: u32) -> Self {
        Self {
            channels: 2,
            sampling_rate: 44_100,
            num_samples,
            length_ms,
        }
    }
}

impl Default for HcaTrackInfo {
    fn default() -> Self {
        Self {
            channels: 2,
            sampling_rate: 44_100,
            num_samples: 5_831_458,
            length_ms: 132_232,
        }
    }
}

/// Options for building a single-track music cue sheet ACB.
#[derive(Debug, Clone)]
struct MusicAcbConfig {
    cue_id: u32,
    memory_awb_id: u16,
    virtual_cue_suffix: Option<String>,
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
}

// ============================================================================
// UTF Table Builder
// ============================================================================

/// Column definition for UTF table building
#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub typ: u8,
    pub flag: u8,
    pub constant_value: Option<Value>,
}

impl ColumnDef {
    pub fn constant(name: impl Into<String>, value: Value) -> Self {
        let typ = match &value {
            Value::U8(_) => COLUMN_TYPE_1BYTE,
            Value::I8(_) => COLUMN_TYPE_1BYTE2,
            Value::U16(_) => COLUMN_TYPE_2BYTE,
            Value::I16(_) => COLUMN_TYPE_2BYTE2,
            Value::U32(_) => COLUMN_TYPE_4BYTE,
            Value::I32(_) => COLUMN_TYPE_4BYTE2,
            Value::U64(_) => COLUMN_TYPE_8BYTE,
            Value::F32(_) => COLUMN_TYPE_FLOAT,
            Value::String(_) => COLUMN_TYPE_STRING,
            Value::Data(_) => COLUMN_TYPE_DATA,
        };
        Self {
            name: name.into(),
            typ,
            flag: COLUMN_FLAG_NAME | COLUMN_FLAG_DEFAULT,
            constant_value: Some(value),
        }
    }

    pub fn per_row(name: impl Into<String>, typ: u8) -> Self {
        Self {
            name: name.into(),
            typ,
            flag: COLUMN_FLAG_NAME | COLUMN_FLAG_ROW,
            constant_value: None,
        }
    }
}

/// Builder for UTF tables
pub struct UtfTableBuilder {
    pub table_name: String,
    pub columns: Vec<ColumnDef>,
    pub rows: Vec<HashMap<String, Value>>,
    data_offset_alignment: u32,
    table_alignment: u32,
    data_item_alignment: u32,
    encoding: u16,
}

impl UtfTableBuilder {
    pub fn new(table_name: impl Into<String>) -> Self {
        Self {
            table_name: table_name.into(),
            columns: Vec::new(),
            rows: Vec::new(),
            data_offset_alignment: 32,
            table_alignment: 32,
            data_item_alignment: 1,
            encoding: 0,
        }
    }

    pub fn add_column(&mut self, col: ColumnDef) -> &mut Self {
        self.columns.push(col);
        self
    }

    pub fn add_row(&mut self, row: HashMap<String, Value>) -> &mut Self {
        self.rows.push(row);
        self
    }

    pub fn with_data_item_alignment(mut self, alignment: u32) -> Self {
        self.data_item_alignment = alignment.max(1);
        self
    }

    pub fn with_data_offset_alignment(mut self, alignment: u32) -> Self {
        self.data_offset_alignment = alignment.max(1);
        self
    }

    pub fn with_table_alignment(mut self, alignment: u32) -> Self {
        self.table_alignment = alignment.max(1);
        self
    }

    pub fn with_encoding(mut self, encoding: u16) -> Self {
        self.encoding = encoding;
        self
    }

    /// Get the size of a column value in bytes
    fn value_size(typ: u8) -> u32 {
        match typ {
            COLUMN_TYPE_1BYTE | COLUMN_TYPE_1BYTE2 => 1,
            COLUMN_TYPE_2BYTE | COLUMN_TYPE_2BYTE2 => 2,
            COLUMN_TYPE_4BYTE | COLUMN_TYPE_4BYTE2 | COLUMN_TYPE_FLOAT | COLUMN_TYPE_STRING => 4,
            COLUMN_TYPE_8BYTE | COLUMN_TYPE_DATA => 8,
            _ => 0,
        }
    }

    /// Encode string to Shift-JIS with null terminator
    fn encode_string(s: &str) -> Vec<u8> {
        let (encoded, _, _) = SHIFT_JIS.encode(s);
        let mut result = encoded.into_owned();
        result.push(0);
        result
    }

    fn add_value_to_tables(
        value: &Value,
        string_table: &mut StringTable,
        data_table: &mut DataTable,
    ) -> Option<(u32, u32)> {
        match value {
            Value::String(value) => Some((string_table.add(value), 0)),
            Value::Data(value) if !value.is_empty() => {
                Some((data_table.add(value), value.len() as u32))
            }
            _ => None,
        }
    }

    fn collect_column_data(
        &self,
        string_table: &mut StringTable,
        data_table: &mut DataTable,
    ) -> (Vec<u32>, Vec<Option<(u32, u32)>>) {
        let mut names = Vec::with_capacity(self.columns.len());
        let mut values = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            names.push(string_table.add(&column.name));
            values.push(
                column
                    .constant_value
                    .as_ref()
                    .and_then(|value| Self::add_value_to_tables(value, string_table, data_table)),
            );
        }
        (names, values)
    }

    fn collect_row_data(
        &self,
        string_table: &mut StringTable,
        data_table: &mut DataTable,
    ) -> Vec<HashMap<String, (u32, u32)>> {
        let mut rows = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let mut offsets = HashMap::new();
            for column in self
                .columns
                .iter()
                .filter(|column| column.constant_value.is_none())
            {
                let Some(offset) = row
                    .get(&column.name)
                    .and_then(|value| Self::add_value_to_tables(value, string_table, data_table))
                else {
                    continue;
                };
                offsets.insert(column.name.clone(), offset);
            }
            rows.push(offsets);
        }
        rows
    }

    fn schema_size(&self) -> u32 {
        self.columns
            .iter()
            .map(|column| {
                5 + column
                    .constant_value
                    .as_ref()
                    .map_or(0, |_| Self::value_size(column.typ))
            })
            .sum()
    }

    fn write_schema<W: Write>(
        &self,
        writer: &mut W,
        column_name_offsets: &[u32],
        constant_offsets: &[Option<(u32, u32)>],
    ) -> Result<(), BuilderError> {
        for (index, column) in self.columns.iter().enumerate() {
            writer.write_all(&[column.flag | column.typ])?;
            write_u32_be(writer, column_name_offsets[index])?;
            if let Some(value) = &column.constant_value {
                self.write_value(writer, value, &constant_offsets[index])?;
            }
        }
        Ok(())
    }

    fn write_rows<W: Write>(
        &self,
        writer: &mut W,
        row_offsets: &[HashMap<String, (u32, u32)>],
    ) -> Result<(), BuilderError> {
        for (row_index, row) in self.rows.iter().enumerate() {
            for column in self
                .columns
                .iter()
                .filter(|column| column.constant_value.is_none())
            {
                if let Some(value) = row.get(&column.name) {
                    let offset = row_offsets[row_index].get(&column.name).copied();
                    self.write_value(writer, value, &offset)?;
                } else {
                    self.write_default(writer, column.typ)?;
                }
            }
        }
        Ok(())
    }

    /// Build the UTF table and write to output
    pub fn build<W: Write + Seek>(&self, writer: &mut W) -> Result<(), BuilderError> {
        // Phase 1: Collect all strings and data blobs
        let mut string_table = StringTable::new();
        let mut data_table = DataTable::new(self.data_item_alignment);

        // Add table name
        let table_name_offset = string_table.add_table_name(&self.table_name);

        let (column_name_offsets, constant_offsets) =
            self.collect_column_data(&mut string_table, &mut data_table);
        let row_string_data_offsets = self.collect_row_data(&mut string_table, &mut data_table);

        // Phase 2: Calculate sizes and offsets
        // Schema size: for each column, 1 byte (flag|type) + 4 bytes (name offset) + optional constant value
        let schema_size = self.schema_size();

        // Row size: sum of per-row column sizes
        let row_width: u32 = self
            .columns
            .iter()
            .filter(|c| c.constant_value.is_none())
            .map(|c| Self::value_size(c.typ))
            .sum();

        let rows_size = row_width * self.rows.len() as u32;

        // Offsets are calculated as absolute table positions. CRI stores them
        // relative to byte 8, after the @UTF magic and table-size field.
        let schema_offset: u32 = 0x20;
        let rows_offset = schema_offset + schema_size;
        let strings_offset = rows_offset + rows_size;
        let unaligned_data_offset = strings_offset + string_table.len() as u32;
        let data_offset = align_to(unaligned_data_offset, self.data_offset_alignment);
        let table_size = align_to(data_offset + data_table.len() as u32, self.table_alignment);

        // Phase 3: Write header
        writer.write_all(b"@UTF")?;
        write_u32_be(writer, table_size - 8)?; // table_size (excluding magic and this field)
        write_u16_be(writer, self.encoding)?; // unknown byte + character encoding
        write_u16_be(writer, (rows_offset - 8) as u16)?; // rows_offset relative to +8
        write_u32_be(writer, strings_offset - 8)?; // strings_offset relative to +8
        write_u32_be(writer, data_offset - 8)?; // data_offset relative to +8
        write_u32_be(writer, table_name_offset)?; // table_name offset in string table
        write_u16_be(writer, self.columns.len() as u16)?; // number of columns
        write_u16_be(writer, row_width as u16)?; // row width
        write_u32_be(writer, self.rows.len() as u32)?; // number of rows

        // Phase 4: Write schema
        self.write_schema(writer, &column_name_offsets, &constant_offsets)?;

        // Phase 5: Write rows
        self.write_rows(writer, &row_string_data_offsets)?;

        // Phase 6: Write string table
        writer.write_all(string_table.data())?;
        write_padding(writer, (data_offset - unaligned_data_offset) as usize)?;

        // Phase 7: Write data table
        writer.write_all(data_table.data())?;
        write_padding(
            writer,
            (table_size - (data_offset + data_table.len() as u32)) as usize,
        )?;

        Ok(())
    }

    fn write_value<W: Write>(
        &self,
        writer: &mut W,
        val: &Value,
        offset_pair: &Option<(u32, u32)>,
    ) -> Result<(), BuilderError> {
        match val {
            Value::U8(v) => writer.write_all(&[*v])?,
            Value::I8(v) => writer.write_all(&[*v as u8])?,
            Value::U16(v) => write_u16_be(writer, *v)?,
            Value::I16(v) => write_i16_be(writer, *v)?,
            Value::U32(v) => write_u32_be(writer, *v)?,
            Value::I32(v) => write_i32_be(writer, *v)?,
            Value::U64(v) => write_u64_be(writer, *v)?,
            Value::F32(v) => write_f32_be(writer, *v)?,
            Value::String(_) => {
                if let Some((offset, _)) = offset_pair {
                    write_u32_be(writer, *offset)?;
                } else {
                    write_u32_be(writer, 0)?;
                }
            }
            Value::Data(_) => {
                if let Some((offset, size)) = offset_pair {
                    write_u32_be(writer, *offset)?;
                    write_u32_be(writer, *size)?;
                } else {
                    write_u32_be(writer, 0)?;
                    write_u32_be(writer, 0)?;
                }
            }
        }
        Ok(())
    }

    fn write_default<W: Write>(&self, writer: &mut W, typ: u8) -> Result<(), BuilderError> {
        let size = Self::value_size(typ) as usize;
        writer.write_all(&vec![0u8; size])?;
        Ok(())
    }
}

// ============================================================================
// String Table
// ============================================================================

struct StringTable {
    data: Vec<u8>,
    offsets: HashMap<String, u32>,
}

impl StringTable {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets: HashMap::new(),
        }
    }

    fn add(&mut self, s: &str) -> u32 {
        if s.is_empty() {
            return self.add_table_name(s);
        }
        if let Some(&offset) = self.offsets.get(s) {
            return offset;
        }
        let offset = self.data.len() as u32;
        let encoded = UtfTableBuilder::encode_string(s);
        self.data.extend(encoded);
        self.offsets.insert(s.to_string(), offset);
        offset
    }

    fn add_table_name(&mut self, s: &str) -> u32 {
        let offset = self.data.len() as u32;
        let encoded = UtfTableBuilder::encode_string(s);
        self.data.extend(encoded);
        offset
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn data(&self) -> &[u8] {
        &self.data
    }
}

// ============================================================================
// Data Table
// ============================================================================

struct DataTable {
    data: Vec<u8>,
    alignment: u32,
}

impl DataTable {
    fn new(alignment: u32) -> Self {
        Self {
            data: Vec::new(),
            alignment,
        }
    }

    fn add(&mut self, d: &[u8]) -> u32 {
        let aligned_len = align_to(self.data.len() as u32, self.alignment) as usize;
        if aligned_len > self.data.len() {
            self.data.resize(aligned_len, 0);
        }
        let offset = aligned_len as u32;
        self.data.extend(d);
        offset
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn data(&self) -> &[u8] {
        &self.data
    }
}

// ============================================================================
// AFS2 Archive Builder
// ============================================================================

/// Builder for AFS2 (AWB) archives
pub struct AfsArchiveBuilder {
    alignment: u32,
    subkey: u16,
    files: Vec<(u32, Vec<u8>)>, // (cue_id, data)
}

impl AfsArchiveBuilder {
    pub fn new() -> Self {
        Self {
            alignment: 32,
            subkey: 0,
            files: Vec::new(),
        }
    }

    pub fn with_alignment(mut self, alignment: u32) -> Self {
        self.alignment = alignment;
        self
    }

    pub fn with_subkey(mut self, subkey: u16) -> Self {
        self.subkey = subkey;
        self
    }

    pub fn add_file(&mut self, cue_id: u32, data: Vec<u8>) -> &mut Self {
        self.files.push((cue_id, data));
        self
    }

    /// Build the AFS2 archive and write to output
    pub fn build<W: Write + Seek>(&self, writer: &mut W) -> Result<(), BuilderError> {
        if self.files.is_empty() {
            return Err(BuilderError::NoTracks);
        }

        let ordered_files = self.ordered_files();
        let file_count = ordered_files.len() as u32;
        let id_field_len = self.id_field_len(&ordered_files);
        let position_field_len = self.position_field_len(id_field_len, &ordered_files);
        let header_size = 16 + id_field_len * file_count + position_field_len * (file_count + 1);
        let file_offsets = self.file_offsets(header_size, &ordered_files);

        // Write header
        writer.write_all(b"AFS2")?;
        write_u32_le(
            writer,
            (if self.subkey != 0 { 2 } else { 1 })
                | (position_field_len << 8)
                | (id_field_len << 16),
        )?;
        write_u32_le(writer, file_count)?;
        write_u16_le(writer, self.alignment as u16)?;
        write_u16_le(writer, self.subkey)?;

        // Write file IDs
        for (cue_id, _) in &ordered_files {
            write_sized_le(writer, *cue_id as u64, id_field_len)?;
        }

        // Write offset table
        for offset in &file_offsets {
            write_sized_le(writer, *offset, position_field_len)?;
        }

        // Write file data with alignment padding
        for (_, data) in &ordered_files {
            let current_pos = writer.stream_position()? as u32;
            let aligned_pos = align_to(current_pos, self.alignment);
            write_padding(writer, (aligned_pos - current_pos) as usize)?;
            writer.write_all(data)?;
        }

        Ok(())
    }

    fn ordered_files(&self) -> Vec<(u32, Vec<u8>)> {
        let mut files = self.files.clone();
        files.sort_by_key(|(cue_id, _)| *cue_id);
        files
    }

    fn id_field_len(&self, files: &[(u32, Vec<u8>)]) -> u32 {
        let max_id = files.iter().map(|(cue_id, _)| *cue_id).max().unwrap_or(0);
        if files.len() <= u16::MAX as usize && max_id <= u16::MAX as u32 {
            2
        } else {
            4
        }
    }

    fn position_field_len(&self, id_field_len: u32, files: &[(u32, Vec<u8>)]) -> u32 {
        let file_count = files.len() as u32;
        let header_size = 16 + id_field_len * file_count + 2 * (file_count + 1);
        let end_offset = self
            .file_offsets(header_size, files)
            .last()
            .copied()
            .unwrap_or(0);

        if end_offset <= u16::MAX as u64 {
            2
        } else {
            4
        }
    }

    fn file_offsets(&self, header_size: u32, files: &[(u32, Vec<u8>)]) -> Vec<u64> {
        let mut offsets = Vec::with_capacity(files.len() + 1);
        let mut current_offset = header_size as u64;

        for (_, data) in files {
            offsets.push(current_offset);
            current_offset =
                align_to_u64(current_offset, self.alignment as u64) + data.len() as u64;
        }

        offsets.push(current_offset);
        offsets
    }
}

impl Default for AfsArchiveBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ACB Builder
// ============================================================================

/// Builder for complete ACB files
pub struct AcbBuilder {
    tracks: Vec<TrackInput>,
    acb_version: u32,
    streaming_awb: bool,
    music_acb_config: Option<MusicAcbConfig>,
}

impl AcbBuilder {
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            acb_version: 0x01300500, // Common ACB version
            streaming_awb: false,
            music_acb_config: None,
        }
    }

    pub fn add_track(&mut self, track: TrackInput) -> &mut Self {
        self.tracks.push(track);
        self
    }

    pub fn streaming_awb(mut self, streaming: bool) -> Self {
        self.streaming_awb = streaming;
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn music_acb(
        mut self,
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
    ) -> Self {
        self.music_acb_config = Some(MusicAcbConfig {
            cue_id,
            memory_awb_id,
            virtual_cue_suffix,
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
        });
        self
    }

    /// Build the ACB file and optionally AWB file
    pub fn build<W: Write + Seek>(
        &self,
        acb_writer: &mut W,
        awb_writer: Option<&mut W>,
    ) -> Result<(), BuilderError> {
        if self.tracks.is_empty() {
            return Err(BuilderError::NoTracks);
        }
        self.validate_tracks()?;

        if let Some(config) = &self.music_acb_config {
            return self.build_music_acb(acb_writer, config);
        }

        // Build AWB archive (embedded or external)
        let mut awb_data = Vec::new();
        {
            let mut awb_cursor = std::io::Cursor::new(&mut awb_data);
            let mut awb_builder = AfsArchiveBuilder::new();
            for track in &self.tracks {
                awb_builder.add_file(track.cue_id, track.data.clone());
            }
            awb_builder.build(&mut awb_cursor)?;
        }

        if self.streaming_awb {
            // External AWB: write to separate file
            if let Some(awb_out) = awb_writer {
                awb_out.write_all(&awb_data)?;
            }
        }

        // Build ACB UTF table
        let mut acb_table = UtfTableBuilder::new("Header");

        // Add ACB header columns
        acb_table.add_column(ColumnDef::constant("FileIdentifier", Value::U32(0)));
        acb_table.add_column(ColumnDef::constant("Size", Value::U32(0))); // Will be updated
        acb_table.add_column(ColumnDef::constant("Version", Value::U32(self.acb_version)));
        acb_table.add_column(ColumnDef::constant("Type", Value::U8(0)));
        acb_table.add_column(ColumnDef::constant("Target", Value::U8(0)));
        acb_table.add_column(ColumnDef::constant("AcbVolume", Value::F32(1.0)));
        acb_table.add_column(ColumnDef::constant(
            "NumCueLimit",
            Value::U16(self.tracks.len() as u16),
        ));

        // Add Cue table
        let cue_table = self.build_cue_table()?;
        acb_table.add_column(ColumnDef::constant("CueTable", Value::Data(cue_table)));

        // Add CueName table
        let cue_name_table = self.build_cue_name_table()?;
        acb_table.add_column(ColumnDef::constant(
            "CueNameTable",
            Value::Data(cue_name_table),
        ));

        // Add Sequence/Track/Event tables
        let sequence_table = self.build_sequence_table()?;
        acb_table.add_column(ColumnDef::constant(
            "SequenceTable",
            Value::Data(sequence_table),
        ));

        let track_table = self.build_track_table()?;
        acb_table.add_column(ColumnDef::constant("TrackTable", Value::Data(track_table)));

        let track_event_table = self.build_track_event_table()?;
        acb_table.add_column(ColumnDef::constant(
            "TrackEventTable",
            Value::Data(track_event_table),
        ));

        // Add Waveform table
        let waveform_table = self.build_waveform_table()?;
        acb_table.add_column(ColumnDef::constant(
            "WaveformTable",
            Value::Data(waveform_table),
        ));

        // Add Synth table
        let synth_table = self.build_synth_table()?;
        acb_table.add_column(ColumnDef::constant("SynthTable", Value::Data(synth_table)));

        // Add AWB (embedded or reference)
        if !self.streaming_awb {
            acb_table.add_column(ColumnDef::constant("AwbFile", Value::Data(awb_data)));
        } else {
            acb_table.add_column(ColumnDef::constant(
                "StreamAwbHash",
                Value::Data(vec![0u8; 16]),
            ));
        }

        // Add one row (ACB has single-row header table)
        acb_table.add_row(HashMap::new());

        // Build ACB
        acb_table.build(acb_writer)?;

        Ok(())
    }

    fn build_music_acb<W: Write + Seek>(
        &self,
        acb_writer: &mut W,
        config: &MusicAcbConfig,
    ) -> Result<(), BuilderError> {
        if self.tracks.len() != 1 {
            return Err(BuilderError::MusicAcbRequiresSingleTrack(self.tracks.len()));
        }
        Self::validate_fixed_data("acf_md5_hash", &config.acf_md5_hash, 16)?;
        Self::validate_fixed_data("acb_guid", &config.acb_guid, 16)?;

        let track = &self.tracks[0];
        let info = HcaTrackInfo::from_hca_with_reference(
            &track.data,
            config.reference_num_samples,
            config.reference_length_ms,
        );

        let mut awb_data = Vec::new();
        {
            let mut awb_cursor = std::io::Cursor::new(&mut awb_data);
            let mut awb_builder = AfsArchiveBuilder::new();
            awb_builder.add_file(config.memory_awb_id as u32, track.data.clone());
            awb_builder.build(&mut awb_cursor)?;
        }

        let header = self.build_music_row_header_table(track, info, awb_data, config)?;
        acb_writer.write_all(&header)?;
        Ok(())
    }

    fn build_music_row_header_table(
        &self,
        track: &TrackInput,
        info: HcaTrackInfo,
        awb_data: Vec<u8>,
        config: &MusicAcbConfig,
    ) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("Header").with_data_item_alignment(32);
        let mut row = HashMap::new();

        Self::add_row_value(
            &mut table,
            &mut row,
            "FileIdentifier",
            COLUMN_TYPE_4BYTE,
            Value::U32(0),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "Size",
            COLUMN_TYPE_4BYTE,
            Value::U32(0),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "Version",
            COLUMN_TYPE_4BYTE,
            Value::U32(config.acb_version),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "Type",
            COLUMN_TYPE_1BYTE,
            Value::U8(0),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "Target",
            COLUMN_TYPE_1BYTE,
            Value::U8(0),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "AcfMd5Hash",
            COLUMN_TYPE_DATA,
            Value::Data(config.acf_md5_hash.clone()),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "CategoryExtension",
            COLUMN_TYPE_1BYTE,
            Value::U8(config.category_extension),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "CueTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_cue_table(info, config)?),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "CueNameTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_cue_name_table(track, config)?),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "WaveformTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_waveform_table(track, info, config)?),
        );
        for name in [
            "AisacTable",
            "GraphTable",
            "GlobalAisacReferenceTable",
            "AisacNameTable",
        ] {
            Self::add_row_value(
                &mut table,
                &mut row,
                name,
                COLUMN_TYPE_DATA,
                Value::Data(Vec::new()),
            );
        }
        Self::add_row_value(
            &mut table,
            &mut row,
            "SynthTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_synth_table(config)?),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "SeqCommandTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_seq_command_table(config)?),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "TrackTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_track_table(config)?),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "SequenceTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_sequence_table(config)?),
        );
        for name in [
            "AisacControlNameTable",
            "AutoModulationTable",
            "StreamAwbTocWorkOld",
        ] {
            Self::add_row_value(
                &mut table,
                &mut row,
                name,
                COLUMN_TYPE_DATA,
                Value::Data(Vec::new()),
            );
        }
        Self::add_row_value(
            &mut table,
            &mut row,
            "AwbFile",
            COLUMN_TYPE_DATA,
            Value::Data(awb_data),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "VersionString",
            COLUMN_TYPE_STRING,
            Value::String(config.version_string.clone()),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "CueLimitWorkTable",
            COLUMN_TYPE_DATA,
            Value::Data(Vec::new()),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "NumCueLimitListWorks",
            COLUMN_TYPE_2BYTE,
            Value::U16(0),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "NumCueLimitNodeWorks",
            COLUMN_TYPE_2BYTE,
            Value::U16(0),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "AcbGuid",
            COLUMN_TYPE_DATA,
            Value::Data(config.acb_guid.clone()),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "StreamAwbHash",
            COLUMN_TYPE_DATA,
            Value::Data(vec![0; 16]),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "StreamAwbTocWork_Old",
            COLUMN_TYPE_DATA,
            Value::Data(Vec::new()),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "AcbVolume",
            COLUMN_TYPE_FLOAT,
            Value::F32(config.acb_volume),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "StringValueTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_string_value_table(config)?),
        );
        for name in ["OutsideLinkTable", "BlockSequenceTable", "BlockTable"] {
            Self::add_row_value(
                &mut table,
                &mut row,
                name,
                COLUMN_TYPE_DATA,
                Value::Data(Vec::new()),
            );
        }
        Self::add_row_value(
            &mut table,
            &mut row,
            "Name",
            COLUMN_TYPE_STRING,
            Value::String(track.name.clone()),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "CharacterEncodingType",
            COLUMN_TYPE_1BYTE,
            Value::U8(0),
        );
        for name in ["EventTable", "ActionTrackTable"] {
            Self::add_row_value(
                &mut table,
                &mut row,
                name,
                COLUMN_TYPE_DATA,
                Value::Data(Vec::new()),
            );
        }
        Self::add_row_value(
            &mut table,
            &mut row,
            "AcfReferenceTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_acf_reference_table(config)?),
        );
        for name in ["WaveformExtensionDataTable", "BeatSyncInfoTable"] {
            Self::add_row_value(
                &mut table,
                &mut row,
                name,
                COLUMN_TYPE_DATA,
                Value::Data(Vec::new()),
            );
        }
        Self::add_row_value(
            &mut table,
            &mut row,
            "CuePriorityType",
            COLUMN_TYPE_1BYTE,
            Value::U8(config.cue_priority_type),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "NumCueLimit",
            COLUMN_TYPE_2BYTE,
            Value::U16(0),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "TrackCommandTable",
            COLUMN_TYPE_DATA,
            Value::Data(Vec::new()),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "SynthCommandTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_synth_command_table()?),
        );
        Self::add_row_value(
            &mut table,
            &mut row,
            "TrackEventTable",
            COLUMN_TYPE_DATA,
            Value::Data(self.build_music_track_event_table(config)?),
        );
        for name in [
            "SeqParameterPalletTable",
            "TrackParameterPalletTable",
            "SynthParameterPalletTable",
            "SoundGeneratorTable",
            "InstrumentPluginTrackTable",
            "InstrumentPluginParameterTable",
        ] {
            Self::add_row_value(
                &mut table,
                &mut row,
                name,
                COLUMN_TYPE_DATA,
                Value::Data(Vec::new()),
            );
        }
        Self::add_row_value(&mut table, &mut row, "R8", COLUMN_TYPE_1BYTE, Value::U8(0));
        Self::add_row_value(
            &mut table,
            &mut row,
            "ProjectKey",
            COLUMN_TYPE_DATA,
            Value::Data(Vec::new()),
        );
        for name in ["R6", "R5", "R4", "R3", "R2", "R1", "R0"] {
            Self::add_row_value(&mut table, &mut row, name, COLUMN_TYPE_1BYTE, Value::U8(0));
        }
        for name in ["PaddingArea", "StreamAwbTocWork", "StreamAwbAfs2Header"] {
            Self::add_row_value(
                &mut table,
                &mut row,
                name,
                COLUMN_TYPE_DATA,
                Value::Data(Vec::new()),
            );
        }

        table.add_row(row);
        Self::finish_table(table)
    }

    fn validate_tracks(&self) -> Result<(), BuilderError> {
        if self.tracks.len() > u16::MAX as usize {
            return Err(BuilderError::TooManyTracks(self.tracks.len()));
        }

        if let Some(config) = &self.music_acb_config {
            let last_cue_id = config
                .cue_id
                .saturating_add(Self::music_cue_count(config) as u32)
                .saturating_sub(1);
            if last_cue_id > u16::MAX as u32 {
                return Err(BuilderError::CueIdTooLarge(last_cue_id));
            }
        }

        let mut cue_ids = HashSet::with_capacity(self.tracks.len());
        for track in &self.tracks {
            if track.cue_id > u16::MAX as u32 {
                return Err(BuilderError::CueIdTooLarge(track.cue_id));
            }
            if !cue_ids.insert(track.cue_id) {
                return Err(BuilderError::DuplicateCueId(track.cue_id));
            }
        }

        Ok(())
    }

    fn finish_table(table: UtfTableBuilder) -> Result<Vec<u8>, BuilderError> {
        let mut buf = std::io::Cursor::new(Vec::new());
        table.build(&mut buf)?;
        Ok(buf.into_inner())
    }

    fn finish_music_nested_table(table: UtfTableBuilder) -> Result<Vec<u8>, BuilderError> {
        Self::finish_table(
            table
                .with_encoding(1)
                .with_data_offset_alignment(1)
                .with_table_alignment(4),
        )
    }

    fn validate_fixed_data(
        field: &'static str,
        data: &[u8],
        expected: usize,
    ) -> Result<(), BuilderError> {
        if data.len() != expected {
            return Err(BuilderError::InvalidFixedDataLength {
                field,
                expected,
                actual: data.len(),
            });
        }
        Ok(())
    }

    fn add_row_value(
        table: &mut UtfTableBuilder,
        row: &mut HashMap<String, Value>,
        name: &'static str,
        typ: u8,
        value: Value,
    ) {
        table.add_column(ColumnDef::per_row(name, typ));
        row.insert(name.to_string(), value);
    }

    fn music_cue_names(track: &TrackInput, config: &MusicAcbConfig) -> Vec<String> {
        let mut names = vec![track.name.clone()];
        if let Some(suffix) = &config.virtual_cue_suffix {
            names.push(format!("{}{}", track.name, suffix));
        }
        names
    }

    fn music_cue_count(config: &MusicAcbConfig) -> u16 {
        if config.virtual_cue_suffix.is_some() {
            2
        } else {
            1
        }
    }

    fn build_music_cue_table(
        &self,
        info: HcaTrackInfo,
        config: &MusicAcbConfig,
    ) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("Cue");

        table.add_column(ColumnDef::per_row("CueId", COLUMN_TYPE_4BYTE));
        table.add_column(ColumnDef::constant("ReferenceType", Value::U8(3)));
        table.add_column(ColumnDef::per_row("ReferenceIndex", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::constant(
            "UserData",
            Value::String(String::new()),
        ));
        table.add_column(ColumnDef::constant("Worksize", Value::U16(0)));
        table.add_column(ColumnDef::constant(
            "AisacControlMap",
            Value::Data(Vec::new()),
        ));
        table.add_column(ColumnDef::constant("Length", Value::U32(info.length_ms)));
        table.add_column(ColumnDef::constant("NumAisacControlMaps", Value::U8(0)));
        table.add_column(ColumnDef::constant("HeaderVisibility", Value::U8(1)));
        table.add_column(ColumnDef::constant("NumRelatedWaveforms", Value::U16(1)));

        for i in 0..Self::music_cue_count(config) {
            let mut row = HashMap::new();
            row.insert(
                "CueId".to_string(),
                Value::U32(config.cue_id.saturating_add(i as u32)),
            );
            row.insert("ReferenceIndex".to_string(), Value::U16(i));
            table.add_row(row);
        }

        Self::finish_music_nested_table(table)
    }

    fn build_music_cue_name_table(
        &self,
        track: &TrackInput,
        config: &MusicAcbConfig,
    ) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("CueName");

        table.add_column(ColumnDef::per_row("CueName", COLUMN_TYPE_STRING));
        table.add_column(ColumnDef::per_row("CueIndex", COLUMN_TYPE_2BYTE));

        for (i, name) in Self::music_cue_names(track, config).into_iter().enumerate() {
            let mut row = HashMap::new();
            row.insert("CueName".to_string(), Value::String(name));
            row.insert("CueIndex".to_string(), Value::U16(i as u16));
            table.add_row(row);
        }

        Self::finish_music_nested_table(table)
    }

    fn build_music_waveform_table(
        &self,
        track: &TrackInput,
        info: HcaTrackInfo,
        config: &MusicAcbConfig,
    ) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("Waveform");

        table.add_column(ColumnDef::per_row("MemoryAwbId", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("EncodeType", COLUMN_TYPE_1BYTE));
        table.add_column(ColumnDef::per_row("Streaming", COLUMN_TYPE_1BYTE));
        table.add_column(ColumnDef::per_row("NumChannels", COLUMN_TYPE_1BYTE));
        table.add_column(ColumnDef::per_row("LoopFlag", COLUMN_TYPE_1BYTE));
        table.add_column(ColumnDef::per_row("SamplingRate", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("NumSamples", COLUMN_TYPE_4BYTE));
        table.add_column(ColumnDef::per_row("ExtensionData", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("StreamAwbPortNo", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("StreamAwbId", COLUMN_TYPE_2BYTE));

        let mut row = HashMap::new();
        row.insert("MemoryAwbId".to_string(), Value::U16(config.memory_awb_id));
        row.insert("EncodeType".to_string(), Value::U8(track.encode_type as u8));
        row.insert("Streaming".to_string(), Value::U8(0));
        row.insert("NumChannels".to_string(), Value::U8(info.channels));
        row.insert("LoopFlag".to_string(), Value::U8(1));
        row.insert("SamplingRate".to_string(), Value::U16(info.sampling_rate));
        row.insert("NumSamples".to_string(), Value::U32(info.num_samples));
        row.insert("ExtensionData".to_string(), Value::U16(0xffff));
        row.insert("StreamAwbPortNo".to_string(), Value::U16(0xffff));
        row.insert("StreamAwbId".to_string(), Value::U16(0xffff));
        table.add_row(row);

        Self::finish_music_nested_table(table)
    }

    fn build_music_synth_table(&self, config: &MusicAcbConfig) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("Synth");

        table.add_column(ColumnDef::constant("Type", Value::U8(0)));
        table.add_column(ColumnDef::constant(
            "VoiceLimitGroupName",
            Value::String(String::new()),
        ));
        table.add_column(ColumnDef::per_row("CommandIndex", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("ReferenceItems", COLUMN_TYPE_DATA));
        table.add_column(ColumnDef::constant("LocalAisacs", Value::Data(Vec::new())));
        table.add_column(ColumnDef::constant(
            "GlobalAisacStartIndex",
            Value::U16(0xffff),
        ));
        table.add_column(ColumnDef::constant("GlobalAisacNumRefs", Value::U16(0)));
        table.add_column(ColumnDef::per_row("ControlWorkArea1", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("ControlWorkArea2", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::constant("TrackValues", Value::Data(Vec::new())));
        table.add_column(ColumnDef::constant("ParameterPallet", Value::U16(0xffff)));
        table.add_column(ColumnDef::constant(
            "ActionTrackStartIndex",
            Value::U16(0xffff),
        ));
        table.add_column(ColumnDef::constant("NumActionTracks", Value::U16(0)));

        for i in 0..Self::music_cue_count(config) {
            let mut row = HashMap::new();
            row.insert(
                "CommandIndex".to_string(),
                Value::U16(if i == 0 { 0xffff } else { 0 }),
            );
            row.insert(
                "ReferenceItems".to_string(),
                Value::Data(vec![0x00, 0x01, 0x00, 0x00]),
            );
            row.insert("ControlWorkArea1".to_string(), Value::U16(i));
            row.insert("ControlWorkArea2".to_string(), Value::U16(i));
            table.add_row(row);
        }

        Self::finish_music_nested_table(table)
    }

    fn build_music_seq_command_table(
        &self,
        config: &MusicAcbConfig,
    ) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("SequenceCommand");
        table.add_column(ColumnDef::per_row("Command", COLUMN_TYPE_DATA));

        let commands = [
            vec![
                0x00, 0x41, 0x04, 0x00, 0x00, 0x00, 0x05, 0x00, 0x45, 0x04, 0x41, 0xf0, 0x00, 0x00,
                0x00, 0x6f, 0x04, 0x00, 0x00, 0x27, 0x10,
            ],
            vec![
                0x00, 0x41, 0x04, 0x00, 0x00, 0x00, 0x05, 0x00, 0x45, 0x04, 0x41, 0xf0, 0x00, 0x00,
                0x00, 0x6f, 0x04, 0x00, 0x00, 0x23, 0x28, 0x00, 0x6f, 0x04, 0x00, 0x01, 0x13, 0x88,
                0x00, 0x6f, 0x04, 0x00, 0x03, 0x13, 0x88,
            ],
        ];
        for command in commands
            .into_iter()
            .take(Self::music_cue_count(config) as usize)
        {
            let mut row = HashMap::new();
            row.insert("Command".to_string(), Value::Data(command));
            table.add_row(row);
        }

        Self::finish_music_nested_table(table)
    }

    fn build_music_track_table(&self, config: &MusicAcbConfig) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("Track");

        table.add_column(ColumnDef::per_row("EventIndex", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::constant("CommandIndex", Value::U16(0xffff)));
        table.add_column(ColumnDef::constant("LocalAisacs", Value::Data(Vec::new())));
        table.add_column(ColumnDef::constant(
            "GlobalAisacStartIndex",
            Value::U16(0xffff),
        ));
        table.add_column(ColumnDef::constant("GlobalAisacNumRefs", Value::U16(0)));
        table.add_column(ColumnDef::constant("ParameterPallet", Value::U16(0xffff)));
        table.add_column(ColumnDef::constant("TargetType", Value::U8(0)));
        table.add_column(ColumnDef::constant(
            "TargetName",
            Value::String(String::new()),
        ));
        table.add_column(ColumnDef::constant("TargetId", Value::U32(u32::MAX)));
        table.add_column(ColumnDef::constant(
            "TargetAcbName",
            Value::String(String::new()),
        ));
        table.add_column(ColumnDef::constant("Scope", Value::U8(0)));
        table.add_column(ColumnDef::constant("TargetTrackNo", Value::U16(0xffff)));

        for i in 0..Self::music_cue_count(config) {
            let mut row = HashMap::new();
            row.insert("EventIndex".to_string(), Value::U16(i));
            table.add_row(row);
        }

        Self::finish_music_nested_table(table)
    }

    fn build_music_sequence_table(&self, config: &MusicAcbConfig) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("Sequence");

        table.add_column(ColumnDef::constant("PlaybackRatio", Value::U16(100)));
        table.add_column(ColumnDef::constant("NumTracks", Value::U16(1)));
        table.add_column(ColumnDef::per_row("TrackIndex", COLUMN_TYPE_DATA));
        table.add_column(ColumnDef::per_row("CommandIndex", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::constant("LocalAisacs", Value::Data(Vec::new())));
        table.add_column(ColumnDef::constant(
            "GlobalAisacStartIndex",
            Value::U16(0xffff),
        ));
        table.add_column(ColumnDef::constant("GlobalAisacNumRefs", Value::U16(0)));
        table.add_column(ColumnDef::constant("ParameterPallet", Value::U16(0xffff)));
        table.add_column(ColumnDef::constant(
            "ActionTrackStartIndex",
            Value::U16(0xffff),
        ));
        table.add_column(ColumnDef::constant("NumActionTracks", Value::U16(0)));
        table.add_column(ColumnDef::constant("TrackValues", Value::Data(Vec::new())));
        table.add_column(ColumnDef::constant("Type", Value::U8(0)));
        table.add_column(ColumnDef::per_row("ControlWorkArea1", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("ControlWorkArea2", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::constant(
            "InstPluginTrackStartIndex",
            Value::U16(0xffff),
        ));
        table.add_column(ColumnDef::constant("NumInstPluginTracks", Value::U16(0)));

        for i in 0..Self::music_cue_count(config) {
            let mut row = HashMap::new();
            row.insert(
                "TrackIndex".to_string(),
                Value::Data(i.to_be_bytes().to_vec()),
            );
            row.insert("CommandIndex".to_string(), Value::U16(i));
            row.insert("ControlWorkArea1".to_string(), Value::U16(i));
            row.insert("ControlWorkArea2".to_string(), Value::U16(i));
            table.add_row(row);
        }

        Self::finish_music_nested_table(table)
    }

    fn build_music_string_value_table(
        &self,
        config: &MusicAcbConfig,
    ) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("Strings");
        table.add_column(ColumnDef::per_row("StringValue", COLUMN_TYPE_STRING));

        for value in &config.acf_bus_names {
            let mut row = HashMap::new();
            row.insert("StringValue".to_string(), Value::String(value.clone()));
            table.add_row(row);
        }

        Self::finish_music_nested_table(table)
    }

    fn build_music_acf_reference_table(
        &self,
        config: &MusicAcbConfig,
    ) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("AcfReference");
        table.add_column(ColumnDef::per_row("Type", COLUMN_TYPE_1BYTE));
        table.add_column(ColumnDef::per_row("Name", COLUMN_TYPE_STRING));
        table.add_column(ColumnDef::constant("Name2", Value::String(String::new())));
        table.add_column(ColumnDef::per_row("Id", COLUMN_TYPE_4BYTE));

        let mut category = HashMap::new();
        category.insert("Type".to_string(), Value::U8(3));
        category.insert(
            "Name".to_string(),
            Value::String(config.acf_category_name.clone()),
        );
        category.insert("Id".to_string(), Value::U32(config.acf_category_id));
        table.add_row(category);

        for name in &config.acf_bus_names {
            let mut row = HashMap::new();
            row.insert("Type".to_string(), Value::U8(9));
            row.insert("Name".to_string(), Value::String(name.clone()));
            row.insert("Id".to_string(), Value::U32(u32::MAX));
            table.add_row(row);
        }

        Self::finish_music_nested_table(table)
    }

    fn build_music_synth_command_table(&self) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("SynthCommand");
        table.add_column(ColumnDef::per_row("Command", COLUMN_TYPE_DATA));

        let mut row = HashMap::new();
        row.insert(
            "Command".to_string(),
            Value::Data(vec![0x00, 0x6f, 0x04, 0x00, 0x03, 0x13, 0x88]),
        );
        table.add_row(row);

        Self::finish_music_nested_table(table)
    }

    fn build_music_track_event_table(
        &self,
        config: &MusicAcbConfig,
    ) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("TrackEvent");
        table.add_column(ColumnDef::per_row("Command", COLUMN_TYPE_DATA));

        for i in 0..Self::music_cue_count(config) {
            let mut row = HashMap::new();
            row.insert(
                "Command".to_string(),
                Value::Data(Self::synth_command(i as usize)),
            );
            table.add_row(row);
        }

        Self::finish_music_nested_table(table)
    }

    fn build_cue_table(&self) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("CueTable");

        table.add_column(ColumnDef::per_row("CueId", COLUMN_TYPE_4BYTE));
        table.add_column(ColumnDef::per_row("ReferenceType", COLUMN_TYPE_1BYTE));
        table.add_column(ColumnDef::per_row("ReferenceIndex", COLUMN_TYPE_2BYTE));

        for (i, track) in self.tracks.iter().enumerate() {
            let mut row = HashMap::new();
            row.insert("CueId".to_string(), Value::U32(track.cue_id));
            row.insert("ReferenceType".to_string(), Value::U8(3)); // Sequence reference
            row.insert("ReferenceIndex".to_string(), Value::U16(i as u16));
            table.add_row(row);
        }

        let mut buf = std::io::Cursor::new(Vec::new());
        table.build(&mut buf)?;
        Ok(buf.into_inner())
    }

    fn build_cue_name_table(&self) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("CueNameTable");

        table.add_column(ColumnDef::per_row("CueIndex", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("CueName", COLUMN_TYPE_STRING));

        for (i, track) in self.tracks.iter().enumerate() {
            let mut row = HashMap::new();
            row.insert("CueIndex".to_string(), Value::U16(i as u16));
            row.insert("CueName".to_string(), Value::String(track.name.clone()));
            table.add_row(row);
        }

        let mut buf = std::io::Cursor::new(Vec::new());
        table.build(&mut buf)?;
        Ok(buf.into_inner())
    }

    fn build_sequence_table(&self) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("SequenceTable");

        table.add_column(ColumnDef::per_row("Type", COLUMN_TYPE_1BYTE));
        table.add_column(ColumnDef::per_row("NumTracks", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("TrackIndex", COLUMN_TYPE_DATA));

        for i in 0..self.tracks.len() {
            let mut row = HashMap::new();
            row.insert("Type".to_string(), Value::U8(0)); // Polyphonic
            row.insert("NumTracks".to_string(), Value::U16(1));
            row.insert(
                "TrackIndex".to_string(),
                Value::Data((i as u16).to_be_bytes().to_vec()),
            );
            table.add_row(row);
        }

        let mut buf = std::io::Cursor::new(Vec::new());
        table.build(&mut buf)?;
        Ok(buf.into_inner())
    }

    fn build_track_table(&self) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("TrackTable");

        table.add_column(ColumnDef::per_row("EventIndex", COLUMN_TYPE_2BYTE));

        for i in 0..self.tracks.len() {
            let mut row = HashMap::new();
            row.insert("EventIndex".to_string(), Value::U16(i as u16));
            table.add_row(row);
        }

        let mut buf = std::io::Cursor::new(Vec::new());
        table.build(&mut buf)?;
        Ok(buf.into_inner())
    }

    fn build_track_event_table(&self) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("TrackEventTable");

        table.add_column(ColumnDef::per_row("Command", COLUMN_TYPE_DATA));

        for i in 0..self.tracks.len() {
            let mut row = HashMap::new();
            row.insert("Command".to_string(), Value::Data(Self::synth_command(i)));
            table.add_row(row);
        }

        let mut buf = std::io::Cursor::new(Vec::new());
        table.build(&mut buf)?;
        Ok(buf.into_inner())
    }

    fn synth_command(synth_index: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(10);
        data.extend_from_slice(&0x07d0u16.to_be_bytes());
        data.push(4);
        data.extend_from_slice(&2u16.to_be_bytes());
        data.extend_from_slice(&(synth_index as u16).to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes());
        data.push(0);
        data
    }

    fn build_waveform_table(&self) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("WaveformTable");

        table.add_column(ColumnDef::per_row("Id", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("EncodeType", COLUMN_TYPE_1BYTE));
        table.add_column(ColumnDef::per_row("Streaming", COLUMN_TYPE_1BYTE));
        table.add_column(ColumnDef::per_row("MemoryAwbId", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("StreamAwbId", COLUMN_TYPE_2BYTE));
        table.add_column(ColumnDef::per_row("StreamAwbPortNo", COLUMN_TYPE_2BYTE));

        for track in &self.tracks {
            let mut row = HashMap::new();
            row.insert("Id".to_string(), Value::U16(track.cue_id as u16));
            row.insert("EncodeType".to_string(), Value::U8(track.encode_type as u8));
            row.insert(
                "Streaming".to_string(),
                Value::U8(if self.streaming_awb { 1 } else { 0 }),
            );
            row.insert(
                "MemoryAwbId".to_string(),
                Value::U16(if self.streaming_awb {
                    0xFFFF
                } else {
                    track.cue_id as u16
                }),
            );
            row.insert(
                "StreamAwbId".to_string(),
                Value::U16(if self.streaming_awb {
                    track.cue_id as u16
                } else {
                    0xFFFF
                }),
            );
            row.insert(
                "StreamAwbPortNo".to_string(),
                Value::U16(if self.streaming_awb { 0 } else { 0xFFFF }),
            );
            table.add_row(row);
        }

        let mut buf = std::io::Cursor::new(Vec::new());
        table.build(&mut buf)?;
        Ok(buf.into_inner())
    }

    fn build_synth_table(&self) -> Result<Vec<u8>, BuilderError> {
        let mut table = UtfTableBuilder::new("SynthTable");

        table.add_column(ColumnDef::per_row("Type", COLUMN_TYPE_1BYTE));
        table.add_column(ColumnDef::per_row(
            "VoiceLimitGroupName",
            COLUMN_TYPE_STRING,
        ));
        table.add_column(ColumnDef::per_row("ReferenceItems", COLUMN_TYPE_DATA));

        for i in 0..self.tracks.len() {
            let mut row = HashMap::new();
            row.insert("Type".to_string(), Value::U8(0)); // Single waveform
            row.insert(
                "VoiceLimitGroupName".to_string(),
                Value::String(String::new()),
            );

            // ReferenceItems: 2-byte waveform item type + 2-byte waveform index.
            let ref_items = {
                let mut data = Vec::new();
                data.extend(&1u16.to_be_bytes()); // waveform item
                data.extend(&(i as u16).to_be_bytes()); // waveform index
                data
            };
            row.insert("ReferenceItems".to_string(), Value::Data(ref_items));
            table.add_row(row);
        }

        let mut buf = std::io::Cursor::new(Vec::new());
        table.build(&mut buf)?;
        Ok(buf.into_inner())
    }
}

impl Default for AcbBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Binary writing helpers (Big Endian)
// ============================================================================

fn write_u16_be<W: Write>(w: &mut W, v: u16) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

fn write_i16_be<W: Write>(w: &mut W, v: i16) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

fn write_u32_be<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

fn write_i32_be<W: Write>(w: &mut W, v: i32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

fn write_u64_be<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

fn write_f32_be<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

// Little Endian helpers for AFS2
fn write_u16_le<W: Write>(w: &mut W, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u32_le<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_sized_le<W: Write>(w: &mut W, v: u64, size: u32) -> io::Result<()> {
    match size {
        2 => write_u16_le(w, v as u16),
        4 => write_u32_le(w, v as u32),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unsupported AFS2 integer size",
        )),
    }
}

fn align_to(offset: u32, alignment: u32) -> u32 {
    if alignment == 0 {
        offset
    } else {
        offset.div_ceil(alignment) * alignment
    }
}

fn align_to_u64(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        offset
    } else {
        offset.div_ceil(alignment) * alignment
    }
}

fn write_padding<W: Write>(w: &mut W, len: usize) -> io::Result<()> {
    if len > 0 {
        w.write_all(&vec![0u8; len])?;
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_table() {
        let mut st = StringTable::new();
        let off1 = st.add("hello");
        let off2 = st.add("world");
        let off3 = st.add("hello"); // Duplicate

        assert_eq!(off1, 0);
        assert_eq!(off3, off1); // Should reuse
        assert!(off2 > off1);
    }

    #[test]
    fn test_afs_builder() {
        let mut builder = AfsArchiveBuilder::new();
        builder.add_file(0, vec![1, 2, 3, 4]);
        builder.add_file(1, vec![5, 6, 7, 8, 9]);

        let mut output = std::io::Cursor::new(Vec::new());
        builder.build(&mut output).unwrap();

        let data = output.into_inner();
        assert_eq!(&data[0..4], b"AFS2");
        assert_eq!(&data[4..8], &[0x01, 0x02, 0x02, 0x00]);
        assert_eq!(&data[12..14], &32u16.to_le_bytes());
        assert_eq!(&data[14..16], &0u16.to_le_bytes());

        let mut archive = crate::acb::afs::AfsArchive::new(std::io::Cursor::new(data)).unwrap();
        assert_eq!(archive.alignment, 32);
        assert_eq!(archive.subkey, 0);
        assert_eq!(
            archive.file_data_for_cue_id(1).unwrap(),
            vec![5, 6, 7, 8, 9]
        );
    }

    #[test]
    fn test_utf_table_builder() {
        let mut builder = UtfTableBuilder::new("TestTable");
        builder.add_column(ColumnDef::constant("Version", Value::U32(1)));
        builder.add_column(ColumnDef::per_row("Name", COLUMN_TYPE_STRING));

        let mut row = HashMap::new();
        row.insert("Name".to_string(), Value::String("test".to_string()));
        builder.add_row(row);

        let mut output = std::io::Cursor::new(Vec::new());
        builder.build(&mut output).unwrap();

        let data = output.into_inner();
        assert_eq!(&data[0..4], b"@UTF");
        assert_eq!(&data[8..10], &[0, 0]);

        let parsed = crate::acb::utf::UtfTable::new(std::io::Cursor::new(data)).unwrap();
        assert_eq!(parsed.name, "TestTable");
        assert_eq!(
            parsed.rows[0].get("Name").and_then(Value::as_string),
            Some("test")
        );
    }

    #[test]
    fn test_utf_data_pool_alignment() {
        let mut builder = UtfTableBuilder::new("TestTable");
        builder.add_column(ColumnDef::constant("Blob", Value::Data(vec![1, 2, 3])));
        builder.add_row(HashMap::new());

        let mut output = std::io::Cursor::new(Vec::new());
        builder.build(&mut output).unwrap();

        let data = output.into_inner();
        let table_size = u32::from_be_bytes(data[4..8].try_into().unwrap()) + 8;
        let data_offset = u32::from_be_bytes(data[16..20].try_into().unwrap()) + 8;

        assert_eq!(data_offset % 32, 0);
        assert_eq!(table_size % 32, 0);
    }
}
