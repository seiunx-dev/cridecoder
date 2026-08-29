//! Binary reader utilities with endianness support

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use std::io::{self, Read, Seek, SeekFrom, Write};

/// Reader wrapper with typed read methods
pub struct Reader<R: Read + Seek> {
    inner: R,
}

impl<R: Read + Seek> Reader<R> {
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> R {
        self.inner
    }

    pub fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }

    pub fn stream_position(&mut self) -> io::Result<u64> {
        self.inner.stream_position()
    }

    // Big-endian reads
    pub fn read_u8(&mut self) -> io::Result<u8> {
        self.inner.read_u8()
    }

    pub fn read_i8(&mut self) -> io::Result<i8> {
        self.inner.read_i8()
    }

    pub fn read_u16(&mut self) -> io::Result<u16> {
        self.inner.read_u16::<BigEndian>()
    }

    pub fn read_i16(&mut self) -> io::Result<i16> {
        self.inner.read_i16::<BigEndian>()
    }

    pub fn read_u32(&mut self) -> io::Result<u32> {
        self.inner.read_u32::<BigEndian>()
    }

    pub fn read_i32(&mut self) -> io::Result<i32> {
        self.inner.read_i32::<BigEndian>()
    }

    pub fn read_u64(&mut self) -> io::Result<u64> {
        self.inner.read_u64::<BigEndian>()
    }

    pub fn read_f32(&mut self) -> io::Result<f32> {
        self.inner.read_f32::<BigEndian>()
    }

    // Little-endian reads
    pub fn read_u16_le(&mut self) -> io::Result<u16> {
        self.inner.read_u16::<LittleEndian>()
    }

    pub fn read_u32_le(&mut self) -> io::Result<u32> {
        self.inner.read_u32::<LittleEndian>()
    }

    /// Read exact number of bytes
    pub fn read_bytes(&mut self, n: usize) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Append exactly `n` bytes from the current position to `out`, reading
    /// directly into the vector's tail (no intermediate allocation or copy).
    pub fn read_into_vec(&mut self, n: usize, out: &mut Vec<u8>) -> io::Result<()> {
        let start = out.len();
        out.resize(start + n, 0);
        match self.inner.read_exact(&mut out[start..]) {
            Ok(()) => Ok(()),
            Err(e) => {
                out.truncate(start);
                Err(e)
            }
        }
    }

    /// Copy exactly `n` bytes from the current position to a writer.
    pub fn copy_to_writer<W: Write>(&mut self, n: u64, writer: &mut W) -> io::Result<u64> {
        let mut limited = self.inner.by_ref().take(n);
        let copied = io::copy(&mut limited, writer)?;
        if copied == n {
            Ok(copied)
        } else {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to copy exact byte count",
            ))
        }
    }

    /// Read bytes at a specific offset, then restore position
    pub fn read_bytes_at(&mut self, n: usize, offset: u64) -> io::Result<Vec<u8>> {
        let pos = self.inner.stream_position()?;
        self.inner.seek(SeekFrom::Start(offset))?;
        let result = self.read_bytes(n);
        self.inner.seek(SeekFrom::Start(pos))?;
        result
    }

    /// Read null-terminated string
    pub fn read_string0(&mut self) -> io::Result<String> {
        let mut buf = Vec::new();
        loop {
            let b = self.inner.read_u8()?;
            if b == 0 {
                break;
            }
            buf.push(b);
        }
        Ok(decode_cri_string(&buf))
    }

    /// Read null-terminated string at offset, then restore position
    pub fn read_string0_at(&mut self, offset: u64) -> io::Result<String> {
        let pos = self.inner.stream_position()?;
        self.inner.seek(SeekFrom::Start(offset))?;
        let result = self.read_string0();
        self.inner.seek(SeekFrom::Start(pos))?;
        result
    }
}

/// Decode a CRI @UTF string. CRI tables store names as UTF-8, Shift-JIS, or
/// UTF-16; try them in that order (PyCriCodecs utf.py) before lossy fallback.
pub fn decode_cri_string(buf: &[u8]) -> String {
    if let Ok(s) = std::str::from_utf8(buf) {
        return s.to_string();
    }
    let (s, _, had_errors) = encoding_rs::SHIFT_JIS.decode(buf);
    if !had_errors {
        return s.into_owned();
    }
    let (s16, _, had_errors16) = encoding_rs::UTF_16LE.decode(buf);
    if !had_errors16 {
        return s16.into_owned();
    }
    String::from_utf8_lossy(buf).into_owned()
}

/// Calculate alignment
pub fn align(alignment: u32, offset: u32) -> u32 {
    if alignment == 0 {
        return offset;
    }
    offset.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_u32() {
        let data = [0x12, 0x34, 0x56, 0x78];
        let mut reader = Reader::new(Cursor::new(data));
        assert_eq!(reader.read_u32().unwrap(), 0x12345678);
    }

    #[test]
    fn test_read_string0() {
        let data = b"hello\0world";
        let mut reader = Reader::new(Cursor::new(data.to_vec()));
        assert_eq!(reader.read_string0().unwrap(), "hello");
    }

    #[test]
    fn test_align() {
        assert_eq!(align(4, 0), 0);
        assert_eq!(align(4, 1), 4);
        assert_eq!(align(4, 4), 4);
        assert_eq!(align(4, 5), 8);
        assert_eq!(align(32, 100), 128);
    }

    #[test]
    fn test_reader_typed_and_positioned_helpers() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x1234u16.to_be_bytes());
        data.extend_from_slice(&(-2i16).to_be_bytes());
        data.extend_from_slice(&0x1122_3344_5566_7788u64.to_be_bytes());
        data.extend_from_slice(&1.5f32.to_be_bytes());
        data.extend_from_slice(&0x5678u16.to_le_bytes());
        data.extend_from_slice(&0x90ab_cdefu32.to_le_bytes());
        data.extend_from_slice(b"name\0");

        let mut reader = Reader::new(Cursor::new(data));
        assert_eq!(reader.read_u16().unwrap(), 0x1234);
        assert_eq!(reader.read_i16().unwrap(), -2);
        assert_eq!(reader.read_u64().unwrap(), 0x1122_3344_5566_7788);
        assert_eq!(reader.read_f32().unwrap(), 1.5);
        assert_eq!(reader.read_u16_le().unwrap(), 0x5678);
        assert_eq!(reader.read_u32_le().unwrap(), 0x90ab_cdef);
        let string_offset = reader.stream_position().unwrap();
        assert_eq!(reader.read_string0_at(string_offset).unwrap(), "name");
        assert_eq!(reader.stream_position().unwrap(), string_offset);
        assert_eq!(reader.read_string0().unwrap(), "name");
        assert_eq!(reader.into_inner().position(), string_offset + 5);
    }

    #[test]
    fn test_reader_copy_and_exact_read_failures() {
        let mut reader = Reader::new(Cursor::new(b"abcdef".to_vec()));
        reader.seek(SeekFrom::Start(2)).unwrap();
        assert_eq!(reader.read_bytes_at(2, 0).unwrap(), b"ab");
        assert_eq!(reader.stream_position().unwrap(), 2);

        let mut appended = vec![b'x'];
        reader.read_into_vec(3, &mut appended).unwrap();
        assert_eq!(appended, b"xcde");
        let before = appended.clone();
        assert!(reader.read_into_vec(3, &mut appended).is_err());
        assert_eq!(appended, before);

        let mut copied = Vec::new();
        let mut reader = Reader::new(Cursor::new(b"copy".to_vec()));
        assert_eq!(reader.copy_to_writer(4, &mut copied).unwrap(), 4);
        assert_eq!(copied, b"copy");
        assert!(reader.copy_to_writer(1, &mut copied).is_err());
    }

    #[test]
    fn test_decode_cri_string_encodings() {
        assert_eq!(decode_cri_string(b"utf8"), "utf8");
        assert_eq!(
            decode_cri_string(&[0x83, 0x65, 0x83, 0x58, 0x83, 0x67]),
            "テスト"
        );
        assert_eq!(decode_cri_string(&[0x41, 0, 0xff, 0xff]), "A\u{ffff}");
        assert_eq!(align(0, 7), 7);
    }
}
