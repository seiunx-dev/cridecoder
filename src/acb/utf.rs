//! UTF table parser (CRI's custom table format)

use crate::acb::consts::{
    COLUMN_FLAG_DEFAULT, COLUMN_FLAG_MASK, COLUMN_FLAG_NAME, COLUMN_FLAG_ROW,
    COLUMN_FLAG_UNDEFINED, COLUMN_TYPE_1BYTE, COLUMN_TYPE_1BYTE2, COLUMN_TYPE_2BYTE,
    COLUMN_TYPE_2BYTE2, COLUMN_TYPE_4BYTE, COLUMN_TYPE_4BYTE2, COLUMN_TYPE_8BYTE, COLUMN_TYPE_DATA,
    COLUMN_TYPE_FLOAT, COLUMN_TYPE_MASK, COLUMN_TYPE_STRING,
};
use crate::reader::Reader;
use rustc_hash::FxHashMap;
use std::io::{Read, Seek, SeekFrom};
use thiserror::Error;

/// Row/constant map for parsed UTF tables. Uses a fast non-cryptographic hasher
/// (FxHash) since the string keys come from trusted binary tables and the
/// per-cell hashing dominates parsing of large tables.
pub type ValueMap = FxHashMap<String, Value>;

#[derive(Error, Debug)]
pub enum UtfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid UTF magic: 0x{0:08X}")]
    BadMagic(u32),
    #[error("UTF table has no columns")]
    NoColumns,
    #[error("UTF schema too large: {0}")]
    SchemaTooLarge(u32),
    #[error("UTF offset out of bounds")]
    OffsetOutOfBounds,
    #[error("Unsupported @UTF version: {0}")]
    UnsupportedVersion(u16),
    #[error("Unknown column flag: 0x{0:02X}")]
    UnknownColumnFlag(u8),
    #[error("Unknown column type: 0x{0:02X}")]
    UnknownColumnType(u8),
    #[error("Field not found: {0}")]
    FieldNotFound(String),
}

/// UTF table header
#[derive(Debug, Clone, Default)]
pub struct UtfHeader {
    pub table_size: u32,
    pub version: u16,
    pub row_offset: u16,
    pub string_table_offset: u32,
    pub data_offset: u32,
    pub table_name_offset: u32,
    pub number_of_fields: u16,
    pub row_size: u16,
    pub number_of_rows: u32,
}

/// Dynamic value type for UTF table columns
#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
    String(String),
    Data(Vec<u8>),
}

impl Value {
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Data(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::U8(v) => Some(*v as i64),
            Value::I8(v) => Some(*v as i64),
            Value::U16(v) => Some(*v as i64),
            Value::I16(v) => Some(*v as i64),
            Value::U32(v) => Some(*v as i64),
            Value::I32(v) => Some(*v as i64),
            Value::U64(v) => Some(*v as i64),
            _ => None,
        }
    }
}

/// Column schema information
#[derive(Debug, Clone)]
struct ColumnSchema {
    flag: u8,
    typ: u8,
    name: String,
    offset: u32, // for ROW: offset within row
}

/// Deferred data references (for constants)
#[derive(Debug, Clone)]
enum Promise {
    Data { offset: u32, size: u32 },
    String { offset: u32 },
}

/// UTF table
#[derive(Debug, Clone)]
pub struct UtfTable {
    pub header: UtfHeader,
    pub name: String,
    pub dynamic_keys: Vec<String>,
    pub constants: ValueMap,
    pub rows: Vec<ValueMap>,
}

const UTF_MAX_SCHEMA_SIZE: u32 = 0x8000;

impl UtfTable {
    /// Parse a UTF table from a reader
    pub fn new<R: Read + Seek>(r: R) -> Result<Self, UtfError> {
        let mut buf = Reader::new(r);

        // Read and validate magic
        let magic = buf.read_u32()?;
        if magic != 0x40555446 {
            // "@UTF"
            return Err(UtfError::BadMagic(magic));
        }

        // Read header fields (all big-endian)
        let header = UtfHeader {
            table_size: buf.read_u32()?,
            version: buf.read_u16()?,
            row_offset: buf.read_u16()?,
            string_table_offset: buf.read_u32()?,
            data_offset: buf.read_u32()?,
            table_name_offset: buf.read_u32()?,
            number_of_fields: buf.read_u16()?,
            row_size: buf.read_u16()?,
            number_of_rows: buf.read_u32()?,
        };

        // Offsets in the header are relative to byte 8 (after magic + table_size)
        let abs_row_offset = header.row_offset as u32 + 8;
        let abs_string_offset = header.string_table_offset + 8;
        let abs_data_offset = header.data_offset + 8;

        // Validation (matching vgmstream cri_utf.c)
        if header.version != 0 && header.version != 1 {
            return Err(UtfError::UnsupportedVersion(header.version));
        }
        let schema_offset: u32 = 0x20;
        let schema_size = abs_row_offset - schema_offset;
        if header.number_of_fields == 0 {
            return Err(UtfError::NoColumns);
        }
        if schema_size >= UTF_MAX_SCHEMA_SIZE {
            return Err(UtfError::SchemaTooLarge(schema_size));
        }
        if abs_row_offset > header.table_size + 8
            || abs_string_offset > header.table_size + 8
            || abs_data_offset > header.table_size + 8
        {
            return Err(UtfError::OffsetOutOfBounds);
        }

        // Read table name
        let table_name =
            buf.read_string0_at(abs_string_offset as u64 + header.table_name_offset as u64)?;

        // Parse schema
        let (schema, dynamic_keys, constants) =
            Self::parse_schema(&mut buf, &header, abs_string_offset, abs_data_offset)?;

        // Read rows
        let rows = Self::read_rows(
            &mut buf,
            &header,
            &schema,
            &constants,
            abs_row_offset,
            abs_string_offset,
            abs_data_offset,
        )?;

        Ok(Self {
            header,
            name: table_name,
            dynamic_keys,
            constants,
            rows,
        })
    }

    fn column_value_size(typ: u8) -> Result<u32, UtfError> {
        match typ {
            COLUMN_TYPE_1BYTE | COLUMN_TYPE_1BYTE2 => Ok(1),
            COLUMN_TYPE_2BYTE | COLUMN_TYPE_2BYTE2 => Ok(2),
            COLUMN_TYPE_4BYTE | COLUMN_TYPE_4BYTE2 | COLUMN_TYPE_FLOAT | COLUMN_TYPE_STRING => {
                Ok(4)
            }
            COLUMN_TYPE_8BYTE | COLUMN_TYPE_DATA => Ok(8),
            _ => Err(UtfError::UnknownColumnType(typ)),
        }
    }

    #[allow(clippy::type_complexity)]
    fn parse_schema<R: Read + Seek>(
        buf: &mut Reader<R>,
        header: &UtfHeader,
        abs_string_offset: u32,
        abs_data_offset: u32,
    ) -> Result<(Vec<ColumnSchema>, Vec<String>, ValueMap), UtfError> {
        buf.seek(SeekFrom::Start(0x20))?;

        let mut dynamic_keys = Vec::new();
        let mut constants = ValueMap::default();
        let mut schema = Vec::with_capacity(header.number_of_fields as usize);
        let mut row_column_offset: u32 = 0;

        for _i in 0..header.number_of_fields {
            let info = buf.read_u8()?;
            let name_offset = buf.read_u32()?;

            let flag = info & COLUMN_FLAG_MASK;
            let typ = info & COLUMN_TYPE_MASK;

            // Validate flags (matching vgmstream)
            if flag == 0 || (flag & COLUMN_FLAG_NAME) == 0 || (flag & COLUMN_FLAG_UNDEFINED) != 0 {
                return Err(UtfError::UnknownColumnFlag(flag));
            }

            // Read column name
            let name = buf.read_string0_at(abs_string_offset as u64 + name_offset as u64)?;

            let val_size = Self::column_value_size(typ)?;

            let mut col = ColumnSchema {
                flag,
                typ,
                name: name.clone(),
                offset: 0,
            };

            // Handle DEFAULT: data is inline in schema area (constant value for all rows)
            if flag & COLUMN_FLAG_DEFAULT != 0 {
                let val = Self::read_column_value(
                    buf,
                    typ,
                    header,
                    abs_string_offset,
                    abs_data_offset,
                    true,
                )?;
                if let Some(v) = Self::resolve_promise_to_value(
                    buf,
                    &val,
                    header,
                    abs_string_offset,
                    abs_data_offset,
                )? {
                    constants.insert(name, v);
                }
            } else if flag & COLUMN_FLAG_ROW != 0 {
                // ROW: data is in per-row area at this offset
                col.offset = row_column_offset;
                row_column_offset += val_size;
                dynamic_keys.push(col.name.clone());
            }
            // NAME-only (flag == 0x10): column exists but has no data

            schema.push(col);
        }

        Ok((schema, dynamic_keys, constants))
    }

    fn read_column_value<R: Read + Seek>(
        buf: &mut Reader<R>,
        type_key: u8,
        _header: &UtfHeader,
        abs_string_offset: u32,
        abs_data_offset: u32,
        is_constant: bool,
    ) -> Result<ValueOrPromise, UtfError> {
        match type_key {
            COLUMN_TYPE_DATA => {
                let offset = buf.read_u32()?;
                let size = buf.read_u32()?;
                if is_constant {
                    Ok(ValueOrPromise::Promise(Promise::Data { offset, size }))
                } else {
                    let data =
                        buf.read_bytes_at(size as usize, (abs_data_offset + offset) as u64)?;
                    Ok(ValueOrPromise::Value(Value::Data(data)))
                }
            }
            COLUMN_TYPE_STRING => {
                let offset = buf.read_u32()?;
                if is_constant {
                    Ok(ValueOrPromise::Promise(Promise::String { offset }))
                } else {
                    let s = buf.read_string0_at((abs_string_offset + offset) as u64)?;
                    Ok(ValueOrPromise::Value(Value::String(s)))
                }
            }
            COLUMN_TYPE_FLOAT => Ok(ValueOrPromise::Value(Value::F32(buf.read_f32()?))),
            COLUMN_TYPE_8BYTE => Ok(ValueOrPromise::Value(Value::U64(buf.read_u64()?))),
            COLUMN_TYPE_4BYTE2 => Ok(ValueOrPromise::Value(Value::I32(buf.read_i32()?))),
            COLUMN_TYPE_4BYTE => Ok(ValueOrPromise::Value(Value::U32(buf.read_u32()?))),
            COLUMN_TYPE_2BYTE2 => Ok(ValueOrPromise::Value(Value::I16(buf.read_i16()?))),
            COLUMN_TYPE_2BYTE => Ok(ValueOrPromise::Value(Value::U16(buf.read_u16()?))),
            COLUMN_TYPE_1BYTE2 => Ok(ValueOrPromise::Value(Value::I8(buf.read_i8()?))),
            COLUMN_TYPE_1BYTE => Ok(ValueOrPromise::Value(Value::U8(buf.read_u8()?))),
            _ => Err(UtfError::UnknownColumnType(type_key)),
        }
    }

    fn resolve_promise_to_value<R: Read + Seek>(
        buf: &mut Reader<R>,
        val: &ValueOrPromise,
        _header: &UtfHeader,
        abs_string_offset: u32,
        abs_data_offset: u32,
    ) -> Result<Option<Value>, UtfError> {
        match val {
            ValueOrPromise::Value(v) => Ok(Some(v.clone())),
            ValueOrPromise::Promise(Promise::Data { offset, size }) => {
                let data = buf.read_bytes_at(*size as usize, (abs_data_offset + offset) as u64)?;
                Ok(Some(Value::Data(data)))
            }
            ValueOrPromise::Promise(Promise::String { offset }) => {
                let s = buf.read_string0_at((abs_string_offset + offset) as u64)?;
                Ok(Some(Value::String(s)))
            }
        }
    }

    fn read_rows<R: Read + Seek>(
        buf: &mut Reader<R>,
        header: &UtfHeader,
        schema: &[ColumnSchema],
        constants: &ValueMap,
        abs_row_offset: u32,
        abs_string_offset: u32,
        abs_data_offset: u32,
    ) -> Result<Vec<ValueMap>, UtfError> {
        let mut rows = Vec::with_capacity(header.number_of_rows as usize);

        for row_idx in 0..header.number_of_rows {
            let mut row = ValueMap::default();

            // Copy resolved constants into every row
            for (k, v) in constants.iter() {
                row.insert(k.clone(), v.clone());
            }

            // Read per-row dynamic fields using pre-parsed schema
            for col in schema.iter() {
                // Skip columns that are not ROW columns
                if col.flag & COLUMN_FLAG_ROW == 0 || col.flag & COLUMN_FLAG_DEFAULT != 0 {
                    continue;
                }

                // Seek to exact position: row start + column offset within row
                let row_start = abs_row_offset as u64 + (row_idx as u64 * header.row_size as u64);
                let field_pos = row_start + col.offset as u64;
                buf.seek(SeekFrom::Start(field_pos))?;

                let val = Self::read_column_value(
                    buf,
                    col.typ,
                    header,
                    abs_string_offset,
                    abs_data_offset,
                    false,
                )?;
                if let Some(v) = Self::resolve_promise_to_value(
                    buf,
                    &val,
                    header,
                    abs_string_offset,
                    abs_data_offset,
                )? {
                    row.insert(col.name.clone(), v);
                }
            }

            rows.push(row);
        }

        Ok(rows)
    }
}

#[derive(Debug, Clone)]
enum ValueOrPromise {
    Value(Value),
    Promise(Promise),
}

/// Helper to get bytes field from a row
pub fn get_bytes_field<'a>(row: &'a ValueMap, key: &str) -> Option<&'a [u8]> {
    row.get(key).and_then(|v| v.as_bytes())
}

/// Move (take ownership of) a bytes field out of a row, leaving an empty
/// placeholder. Avoids cloning large embedded blobs such as the embedded AWB.
pub fn take_bytes_field(row: &mut ValueMap, key: &str) -> Option<Vec<u8>> {
    match row.get_mut(key) {
        Some(Value::Data(d)) => Some(std::mem::take(d)),
        _ => None,
    }
}

/// Helper to get string field from a row
pub fn get_string_field<'a>(row: &'a ValueMap, key: &str) -> Option<&'a str> {
    row.get(key).and_then(|v| v.as_string())
}

/// Helper to get integer field from a row
pub fn get_int_field(row: &ValueMap, key: &str) -> i64 {
    row.get(key).and_then(|v| v.as_int()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_conversions() {
        let v = Value::U32(42);
        assert_eq!(v.as_int(), Some(42));

        let v = Value::String("test".to_string());
        assert_eq!(v.as_string(), Some("test"));

        let v = Value::Data(vec![1, 2, 3]);
        assert_eq!(v.as_bytes(), Some(&[1u8, 2, 3][..]));
    }
}
