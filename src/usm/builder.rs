//! USM container builder - creates CRI USM video containers

use std::io::{self, Seek, SeekFrom, Write};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum UsmBuilderError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("No video stream provided")]
    NoVideoStream,
    #[error("Invalid video data")]
    InvalidVideoData,
}

/// Stream input for USM building
#[derive(Debug, Clone)]
pub struct StreamInput {
    pub data: Vec<u8>,
    pub stream_type: StreamType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    Video, // M2V MPEG2 video
    Audio, // ADX or HCA audio
}

/// Builder for USM containers
///
/// This builder currently writes a minimal structural USM-like container for
/// packaging experiments. It is not yet a full CRI Sofdec muxer and does not
/// guarantee round-tripping through `extract_usm`.
pub struct UsmBuilder {
    filename: String,
    video_stream: Option<Vec<u8>>,
    audio_streams: Vec<Vec<u8>>,
    encryption_key: Option<u64>,
}

impl UsmBuilder {
    pub fn new(filename: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
            video_stream: None,
            audio_streams: Vec::new(),
            encryption_key: None,
        }
    }

    pub fn video(mut self, data: Vec<u8>) -> Self {
        self.video_stream = Some(data);
        self
    }

    pub fn add_audio(&mut self, data: Vec<u8>) -> &mut Self {
        self.audio_streams.push(data);
        self
    }

    pub fn encryption_key(mut self, key: u64) -> Self {
        self.encryption_key = Some(key);
        self
    }

    /// Build the USM file
    pub fn build<W: Write + Seek>(&self, writer: &mut W) -> Result<(), UsmBuilderError> {
        let video_data = self
            .video_stream
            .as_ref()
            .ok_or(UsmBuilderError::NoVideoStream)?;

        // Write CRID header
        self.write_crid_header(writer)?;

        // Write video stream header (@SFV)
        self.write_sfv_header(writer, video_data)?;

        // Write audio stream header (@SFA) if present
        for (i, audio) in self.audio_streams.iter().enumerate() {
            self.write_sfa_header(writer, audio, i as u32)?;
        }

        // Write stream data chunks
        self.write_stream_data(writer, video_data)?;

        Ok(())
    }

    fn write_crid_header<W: Write + Seek>(&self, writer: &mut W) -> Result<(), UsmBuilderError> {
        // CRID signature
        writer.write_all(b"CRID")?;

        // Block size (will be filled later)
        let block_size_pos = writer.stream_position()?;
        write_u32_be(writer, 0)?; // placeholder

        // Padding/flags
        write_u16_be(writer, 0x0001)?; // version
        write_u16_be(writer, 0x0018)?; // data offset

        // Reserved
        writer.write_all(&[0u8; 16])?;

        // Write UTF table for CRID
        self.write_crid_utf_table(writer)?;
        let utf_end = writer.stream_position()?;

        // Update block size
        let block_size = (utf_end - 8) as u32;
        writer.seek(SeekFrom::Start(block_size_pos))?;
        write_u32_be(writer, block_size)?;
        writer.seek(SeekFrom::Start(utf_end))?;

        // Align to 0x20
        let padding = (0x20 - (utf_end % 0x20)) % 0x20;
        writer.write_all(&vec![0u8; padding as usize])?;

        Ok(())
    }

    fn write_crid_utf_table<W: Write + Seek>(&self, writer: &mut W) -> Result<(), UsmBuilderError> {
        // Simple UTF table with filename
        let filename_bytes = self.filename.as_bytes();

        // @UTF header
        writer.write_all(b"@UTF")?;

        // Table size (placeholder, will calculate)
        let table_size_pos = writer.stream_position()?;
        write_u32_be(writer, 0)?;

        // Version
        write_u16_be(writer, 0x0001)?;

        // Calculate offsets
        let schema_size = 5 + 4; // 1 column: flag|type(1) + name_offset(4) + string_offset(4)
        let rows_offset = 0x18 + schema_size as u16;
        let row_size = 4u16; // One string offset

        write_u16_be(writer, rows_offset - 8)?; // rows_offset (relative to +8)

        let strings_offset = rows_offset as u32 + row_size as u32;
        write_u32_be(writer, strings_offset - 8)?; // strings_offset

        // String table: "CRIUSF_DIR_STREAM\0" + "filename\0" + actual_filename
        let string_table_name = b"CRIUSF_DIR_STREAM\0";
        let col_name = b"filename\0";
        let filename_null = [filename_bytes, &[0u8]].concat();

        let data_offset = strings_offset
            + string_table_name.len() as u32
            + col_name.len() as u32
            + filename_null.len() as u32;
        write_u32_be(writer, data_offset - 8)?; // data_offset

        write_u32_be(writer, 0)?; // table_name_offset (points to "CRIUSF_DIR_STREAM")

        write_u16_be(writer, 1)?; // number of columns
        write_u16_be(writer, row_size)?; // row width
        write_u32_be(writer, 1)?; // number of rows

        // Schema: one column "filename" (string, per-row)
        let flag_type = 0x50 | 0x0A; // PERROW | STRING
        writer.write_all(&[flag_type])?;
        write_u32_be(writer, string_table_name.len() as u32)?; // name offset in string table

        // Row data: offset to filename in string table
        let filename_offset = string_table_name.len() as u32 + col_name.len() as u32;
        write_u32_be(writer, filename_offset)?;

        // String table
        writer.write_all(string_table_name)?;
        writer.write_all(col_name)?;
        writer.write_all(&filename_null)?;

        // Update table size
        let end_pos = writer.stream_position()?;
        let table_size = (end_pos - table_size_pos - 4) as u32;
        writer.seek(SeekFrom::Start(table_size_pos))?;
        write_u32_be(writer, table_size)?;
        writer.seek(SeekFrom::Start(end_pos))?;

        Ok(())
    }

    fn write_sfv_header<W: Write + Seek>(
        &self,
        writer: &mut W,
        video_data: &[u8],
    ) -> Result<(), UsmBuilderError> {
        // @SFV (Stream Format Video) chunk
        writer.write_all(b"@SFV")?;

        // Block size placeholder
        let block_size_pos = writer.stream_position()?;
        write_u32_be(writer, 0)?;

        // Flags
        write_u16_be(writer, 0x0001)?;
        write_u16_be(writer, 0x0018)?;

        // Reserved
        writer.write_all(&[0u8; 16])?;

        // Write video format UTF table
        self.write_video_format_utf(writer, video_data)?;

        let end_pos = writer.stream_position()?;
        let block_size = (end_pos - 8 - block_size_pos + 4) as u32;
        writer.seek(SeekFrom::Start(block_size_pos))?;
        write_u32_be(writer, block_size)?;
        writer.seek(SeekFrom::Start(end_pos))?;

        // Align
        let padding = (0x20 - (end_pos % 0x20)) % 0x20;
        writer.write_all(&vec![0u8; padding as usize])?;

        Ok(())
    }

    fn write_video_format_utf<W: Write + Seek>(
        &self,
        writer: &mut W,
        video_data: &[u8],
    ) -> Result<(), UsmBuilderError> {
        // Minimal video format table
        writer.write_all(b"@UTF")?;

        let table_size_pos = writer.stream_position()?;
        write_u32_be(writer, 0)?; // placeholder

        write_u16_be(writer, 0x0001)?; // version

        // Simple table with video info
        let schema_size = 5 * 3; // 3 columns
        let rows_offset = (0x18 + schema_size) as u16;
        let row_size = 12u16; // 3 x u32

        write_u16_be(writer, rows_offset - 8)?;

        let strings_offset = rows_offset as u32 + row_size as u32;
        write_u32_be(writer, strings_offset - 8)?;

        let string_table = b"VIDEO_HDRINFO\0width\0height\0total_frames\0";
        let data_offset = strings_offset + string_table.len() as u32;
        write_u32_be(writer, data_offset - 8)?;

        write_u32_be(writer, 0)?; // table_name_offset
        write_u16_be(writer, 3)?; // 3 columns
        write_u16_be(writer, row_size)?;
        write_u32_be(writer, 1)?; // 1 row

        // Schema
        let offsets = [14u32, 20, 27]; // Column name offsets
        for offset in offsets {
            writer.write_all(&[0x50 | 0x04])?; // PERROW | UINT32
            write_u32_be(writer, offset)?;
        }

        // Row data (dummy values - would need to parse video)
        write_u32_be(writer, 1920)?; // width
        write_u32_be(writer, 1080)?; // height
        write_u32_be(writer, (video_data.len() / 1000) as u32)?; // estimate frames

        // String table
        writer.write_all(string_table)?;

        let end_pos = writer.stream_position()?;
        let table_size = (end_pos - table_size_pos - 4) as u32;
        writer.seek(SeekFrom::Start(table_size_pos))?;
        write_u32_be(writer, table_size)?;
        writer.seek(SeekFrom::Start(end_pos))?;

        Ok(())
    }

    fn write_sfa_header<W: Write + Seek>(
        &self,
        writer: &mut W,
        _audio_data: &[u8],
        stream_id: u32,
    ) -> Result<(), UsmBuilderError> {
        // @SFA (Stream Format Audio) chunk
        writer.write_all(b"@SFA")?;

        let block_size_pos = writer.stream_position()?;
        write_u32_be(writer, 0)?;

        write_u16_be(writer, 0x0001)?;
        write_u16_be(writer, 0x0018)?;

        writer.write_all(&[0u8; 16])?;

        // Audio format UTF table (minimal)
        writer.write_all(b"@UTF")?;
        let table_size_pos2 = writer.stream_position()?;
        write_u32_be(writer, 0)?;

        write_u16_be(writer, 0x0001)?;
        write_u16_be(writer, 0x10)?; // rows_offset
        write_u32_be(writer, 0x14)?; // strings_offset
        write_u32_be(writer, 0x30)?; // data_offset
        write_u32_be(writer, 0)?; // table_name
        write_u16_be(writer, 1)?; // columns
        write_u16_be(writer, 4)?; // row_width
        write_u32_be(writer, 1)?; // rows

        // Schema
        writer.write_all(&[0x50 | 0x04])?;
        write_u32_be(writer, 14)?;

        // Row
        write_u32_be(writer, stream_id)?;

        // Strings
        writer.write_all(b"AUDIO_HDRINFO\0stream_id\0")?;

        let end_pos = writer.stream_position()?;
        let table_size = (end_pos - table_size_pos2 - 4) as u32;
        writer.seek(SeekFrom::Start(table_size_pos2))?;
        write_u32_be(writer, table_size)?;
        writer.seek(SeekFrom::Start(end_pos))?;

        let block_size = (end_pos - block_size_pos - 4) as u32;
        writer.seek(SeekFrom::Start(block_size_pos))?;
        write_u32_be(writer, block_size)?;
        writer.seek(SeekFrom::Start(end_pos))?;

        // Align
        let padding = (0x20 - (end_pos % 0x20)) % 0x20;
        writer.write_all(&vec![0u8; padding as usize])?;

        Ok(())
    }

    fn write_stream_data<W: Write + Seek>(
        &self,
        writer: &mut W,
        video_data: &[u8],
    ) -> Result<(), UsmBuilderError> {
        // Interleave video and audio chunks
        const CHUNK_SIZE: usize = 0x10000; // 64KB chunks

        let mut video_offset = 0usize;
        let mut audio_offsets: Vec<usize> = vec![0; self.audio_streams.len()];

        while video_offset < video_data.len()
            || audio_offsets
                .iter()
                .zip(&self.audio_streams)
                .any(|(o, a)| *o < a.len())
        {
            // Write video chunk
            if video_offset < video_data.len() {
                let chunk_end = (video_offset + CHUNK_SIZE).min(video_data.len());
                let chunk = &video_data[video_offset..chunk_end];
                self.write_video_chunk(writer, chunk, video_offset as u32)?;
                video_offset = chunk_end;
            }

            // Write audio chunks
            for (i, audio) in self.audio_streams.iter().enumerate() {
                if audio_offsets[i] < audio.len() {
                    let chunk_end = (audio_offsets[i] + CHUNK_SIZE).min(audio.len());
                    let chunk = &audio[audio_offsets[i]..chunk_end];
                    self.write_audio_chunk(writer, chunk, i as u32, audio_offsets[i] as u32)?;
                    audio_offsets[i] = chunk_end;
                }
            }
        }

        // Write end markers
        self.write_end_marker(writer)?;
        for _ in 0..self.audio_streams.len() {
            self.write_end_marker(writer)?;
        }

        Ok(())
    }

    fn write_video_chunk<W: Write + Seek>(
        &self,
        writer: &mut W,
        data: &[u8],
        offset: u32,
    ) -> Result<(), UsmBuilderError> {
        writer.write_all(b"@SBV")?; // Stream Block Video
        write_u32_be(writer, (data.len() + 0x18) as u32)?;

        // Header
        write_u16_be(writer, 0x0001)?;
        write_u16_be(writer, 0x0018)?;
        write_u32_be(writer, offset)?; // stream offset
        write_u32_be(writer, 0)?; // padding size
        writer.write_all(&[0u8; 8])?;

        // Data (apply mask if key set)
        let output_data = if let Some(key) = self.encryption_key {
            self.apply_video_mask(data, key)
        } else {
            data.to_vec()
        };
        writer.write_all(&output_data)?;

        // Align to 0x20
        let pos = writer.stream_position()?;
        let padding = (0x20 - (pos % 0x20)) % 0x20;
        writer.write_all(&vec![0u8; padding as usize])?;

        Ok(())
    }

    fn write_audio_chunk<W: Write + Seek>(
        &self,
        writer: &mut W,
        data: &[u8],
        stream_id: u32,
        offset: u32,
    ) -> Result<(), UsmBuilderError> {
        writer.write_all(b"@SBA")?; // Stream Block Audio
        write_u32_be(writer, (data.len() + 0x18) as u32)?;

        write_u16_be(writer, 0x0001)?;
        write_u16_be(writer, 0x0018)?;
        write_u32_be(writer, offset)?;
        write_u32_be(writer, stream_id)?;
        writer.write_all(&[0u8; 8])?;

        // Data (apply mask if key set)
        let output_data = if let Some(key) = self.encryption_key {
            self.apply_audio_mask(data, key)
        } else {
            data.to_vec()
        };
        writer.write_all(&output_data)?;

        let pos = writer.stream_position()?;
        let padding = (0x20 - (pos % 0x20)) % 0x20;
        writer.write_all(&vec![0u8; padding as usize])?;

        Ok(())
    }

    fn write_end_marker<W: Write>(&self, writer: &mut W) -> Result<(), UsmBuilderError> {
        writer.write_all(b"@END")?;
        write_u32_be(writer, 0x18)?;
        write_u16_be(writer, 0x0001)?;
        write_u16_be(writer, 0x0018)?;
        writer.write_all(&[0u8; 16])?;
        Ok(())
    }

    fn apply_video_mask(&self, data: &[u8], key: u64) -> Vec<u8> {
        let (vmask, _) = get_mask(key);
        let mut result = data.to_vec();

        if result.len() >= 0x240 {
            // Apply forward mask for encoding (reverse of decoding)
            let base = 0x40;
            let size = result.len() - base;

            if size >= 0x200 {
                // First pass: update the mask first, then XOR — the exact inverse
                // of the decoder's pass-1 order (extractor.rs mask_video), so an
                // encoded stream round-trips. (Was XOR-then-update; PyCriCodecs usm.py.)
                let mut mask = vmask[0].clone();
                for i in 0..0x100 {
                    mask[i & 0x1F] ^= result[0x100 + base + i];
                    result[base + i] ^= mask[i & 0x1F];
                }

                // Second pass
                let mut mask = vmask[1].clone();
                for i in 0x100..size {
                    let old_val = result[base + i];
                    result[base + i] ^= mask[i & 0x1F];
                    mask[i & 0x1F] = old_val ^ vmask[1][i & 0x1F];
                }
            }
        }

        result
    }

    fn apply_audio_mask(&self, data: &[u8], key: u64) -> Vec<u8> {
        let (_, amask) = get_mask(key);
        let mut result = data.to_vec();

        if result.len() > 0x140 {
            let base = 0x140;
            let size = result.len() - base;
            for i in 0..size {
                result[base + i] ^= amask[i & 0x1F];
            }
        }

        result
    }
}

/// Generate mask from key (same as extractor)
fn get_mask(key: u64) -> (Vec<Vec<u8>>, Vec<u8>) {
    let key1 = (key & 0xFFFFFFFF) as u32;
    let key2 = ((key >> 32) & 0xFFFFFFFF) as u32;

    let mut t = [0u8; 0x20];
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
    let mut vmask1 = vec![0u8; 0x20];
    let mut vmask2 = vec![0u8; 0x20];
    let mut amask = vec![0u8; 0x20];

    for (i, &ti) in t.iter().enumerate() {
        vmask1[i] = ti;
        vmask2[i] = ti ^ 0xFF;
        if i & 1 != 0 {
            amask[i] = t2[(i >> 1) & 3];
        } else {
            amask[i] = ti ^ 0xFF;
        }
    }

    (vec![vmask1, vmask2], amask)
}

// Binary writing helpers
fn write_u16_be<W: Write>(w: &mut W, v: u16) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

fn write_u32_be<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usm_builder_basic() {
        let video_data = vec![0u8; 1000];

        let builder = UsmBuilder::new("test_video").video(video_data);

        let mut output = std::io::Cursor::new(Vec::new());
        builder.build(&mut output).unwrap();

        let data = output.into_inner();
        assert_eq!(&data[0..4], b"CRID");
    }

    #[test]
    fn test_mask_generation() {
        let key: u64 = 0x12345678_9ABCDEF0;
        let (vmask, amask) = get_mask(key);

        assert_eq!(vmask.len(), 2);
        assert_eq!(vmask[0].len(), 0x20);
        assert_eq!(vmask[1].len(), 0x20);
        assert_eq!(amask.len(), 0x20);
    }

    #[test]
    fn test_builder_audio_encryption_and_multiple_chunks() {
        let mut missing_video = std::io::Cursor::new(Vec::new());
        assert!(matches!(
            UsmBuilder::new("missing.usm").build(&mut missing_video),
            Err(UsmBuilderError::NoVideoStream)
        ));

        let video = vec![0x56; 0x1_0020];
        let audio = vec![0x41; 0x1_0040];
        let mut builder = UsmBuilder::new("encrypted.usm")
            .video(video)
            .encryption_key(0x1234_5678_9abc_def0);
        builder.add_audio(audio);

        let mut output = std::io::Cursor::new(Vec::new());
        builder.build(&mut output).unwrap();
        let data = output.into_inner();

        assert_eq!(&data[..4], b"CRID");
        assert!(data.windows(4).any(|window| window == b"@SFA"));
        assert_eq!(
            data.windows(4).filter(|window| *window == b"@SBV").count(),
            2
        );
        assert_eq!(
            data.windows(4).filter(|window| *window == b"@SBA").count(),
            2
        );
        assert_eq!(
            data.windows(4).filter(|window| *window == b"@END").count(),
            2
        );
    }
}
