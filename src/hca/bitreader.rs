//! Bit-level reader for HCA decoding
//!
//! Uses a byte-aligned u32 accumulator with O(1) `peek`/`read`, mirroring the
//! width-specialized bitreaders used by reference C/C++ HCA decoders. A single
//! `peek(n)` reads up to 4 source bytes into a u32, masks off the already
//! consumed low bits, and right-shifts to align — instead of looping per bit.
//!
//! `bits == 0` returns 0 and reads past EOF return 0, matching the prior
//! per-bit `check_bit` semantics.

#![allow(dead_code)]

/// Bit reader for HCA data
pub struct BitReader<'a> {
    data: &'a [u8],
    position: usize, // Current bit position
}

impl<'a> BitReader<'a> {
    /// Create a new BitReader
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// Create BitReader starting at byte offset
    pub fn with_offset(data: &'a [u8], byte_offset: usize) -> Self {
        Self {
            data,
            position: byte_offset * 8,
        }
    }

    /// Get current bit position
    #[inline]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Borrow the underlying data slice. Used by hot loops that run their own
    /// register-resident bit accumulator and then sync `position` back.
    #[inline]
    pub(crate) fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Set bit position
    #[inline]
    pub fn set_position(&mut self, pos: usize) {
        self.position = pos;
    }

    /// Peek at up to 32 bits without advancing position.
    ///
    /// Loads up to 4 bytes (big-endian, zero-padded past EOF) starting at the
    /// current bit's byte, drops the already-consumed low bits, and keeps the
    /// top `bits`. Branch-free on the fast path (>=4 bytes remaining) so it
    /// stays cheap under the hundreds of reads per subframe.
    #[inline]
    pub fn peek(&self, bits: usize) -> u32 {
        if bits == 0 || bits > 32 {
            return 0;
        }

        // Out-of-range reads return a full zero (matches vgmstream clhca.c bitreader),
        // rather than keeping the available high bits and zero-padding the low ones.
        if self.position + bits > self.data.len() * 8 {
            return 0;
        }

        let byte_idx = self.position >> 3;
        let bit_offset = (self.position & 7) as u32; // consumed bits in the leading byte

        // Fast path: at least 4 bytes available from byte_idx. This is by far
        // the common case since HCA blocks are hundreds of bytes and reads
        // happen near the front.
        let chunk = if byte_idx + 4 <= self.data.len() {
            // Safety: bounds checked just above (byte_idx + 4 <= len).
            unsafe {
                let b = self.data.as_ptr().add(byte_idx);
                u32::from_be_bytes([*b, *b.add(1), *b.add(2), *b.add(3)])
            }
        } else {
            // Slow path near EOF: load available bytes, zero-pad the rest.
            self.load_padded(byte_idx)
        };

        // Drop the already-consumed low bits, keep the top `bits`.
        (chunk << bit_offset) >> (32 - bits)
    }

    /// Load up to 4 bytes big-endian starting at `byte_idx`, zero-padded past
    /// EOF. Only used on the slow path when fewer than 4 bytes remain.
    #[inline(never)]
    fn load_padded(&self, byte_idx: usize) -> u32 {
        let mut chunk: u32 = 0;
        for k in 0..4 {
            chunk <<= 8;
            let pos = byte_idx + k;
            if pos < self.data.len() {
                chunk |= self.data[pos] as u32;
            }
        }
        chunk
    }

    /// Read bits and advance position
    #[inline]
    pub fn read(&mut self, bits: usize) -> u32 {
        let value = self.peek(bits);
        self.position += bits;
        value
    }

    /// Read a known-valid bit width and advance position.
    ///
    /// This is used by HCA hot paths where `bits` comes from fixed decoder
    /// tables (0..=12). It keeps the public `read` semantics but avoids the
    /// wider validation branch on every coefficient.
    #[inline(always)]
    pub(crate) fn read_hca_bits(&mut self, bits: usize) -> u32 {
        debug_assert!(bits <= 32);
        if bits == 0 {
            return 0;
        }

        let byte_idx = self.position >> 3;
        let bit_offset = (self.position & 7) as u32;

        let chunk = if byte_idx + 4 <= self.data.len() {
            unsafe {
                let b = self.data.as_ptr().add(byte_idx);
                u32::from_be_bytes([*b, *b.add(1), *b.add(2), *b.add(3)])
            }
        } else {
            self.load_padded(byte_idx)
        };

        self.position += bits;
        (chunk << bit_offset) >> (32 - bits)
    }

    /// Skip bits
    #[inline]
    pub fn skip(&mut self, bits: usize) {
        self.position += bits;
    }

    /// Start a register-resident forward-only accumulator at the current
    /// position. Call `BitAcc::sync` to write the consumed count back.
    #[inline]
    pub(crate) fn acc(&self) -> BitAcc<'a> {
        BitAcc::new(self.data, self.position)
    }

    /// Advance the position by a signed amount, saturating at zero. Used by the
    /// HCA dequantizer where prefix-codebook adjustments may be negative.
    #[inline]
    pub fn advance_signed(&mut self, delta: i32) {
        if delta >= 0 {
            self.position += delta as usize;
        } else {
            self.position = self.position.saturating_sub((-delta) as usize);
        }
    }

    /// Read a single bit
    #[inline]
    pub fn read_bit(&mut self) -> bool {
        let result = self.peek(1) != 0;
        self.position += 1;
        result
    }

    /// Remaining bits available
    #[inline]
    pub fn remaining_bits(&self) -> usize {
        let total_bits = self.data.len() * 8;
        total_bits.saturating_sub(self.position)
    }

    /// Check if there are enough bits remaining
    #[inline]
    pub fn has_bits(&self, bits: usize) -> bool {
        self.remaining_bits() >= bits
    }

    /// Read signed value (with leading bit as sign)
    pub fn read_signed(&mut self, bits: usize) -> i32 {
        if bits == 0 {
            return 0;
        }
        let value = self.read(bits) as i32;
        // Sign extend if highest bit is set
        let sign_bit = 1 << (bits - 1);
        if value & sign_bit != 0 {
            value | ((-1i32) << bits)
        } else {
            value
        }
    }

    /// Read variable-length value with scale bits encoding
    /// Returns (scale_count, base_resolution)
    pub fn read_scale_count(&mut self) -> (u32, u32) {
        let delta_bits = self.read(3);
        if delta_bits == 0 {
            return (0, 0);
        }

        let count = self.read(delta_bits as usize);
        let resolution = self.read(4);
        (count + 1, resolution)
    }

    /// Read off-by-one encoded value
    /// Used in HCA scale factor decoding
    pub fn read_off_by_one(&mut self, bits: usize) -> u32 {
        let value = self.read(bits);
        if value != 0 {
            value + 1
        } else {
            0
        }
    }
}

/// Bit writer for HCA encoding (if needed)
pub struct BitWriter {
    data: Vec<u8>,
    position: usize,
}

impl BitWriter {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0; capacity],
            position: 0,
        }
    }

    pub fn write(&mut self, value: u32, bits: usize) {
        if bits == 0 || bits > 32 {
            return;
        }

        for i in 0..bits {
            let bit = (value >> (bits - 1 - i)) & 1;
            let byte_idx = self.position / 8;
            let bit_idx = 7 - (self.position % 8);

            if byte_idx < self.data.len() {
                if bit != 0 {
                    self.data[byte_idx] |= 1 << bit_idx;
                } else {
                    self.data[byte_idx] &= !(1 << bit_idx);
                }
            }
            self.position += 1;
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }
}

/// Register-resident, forward-only bit accumulator over a byte slice.
///
/// MSB-first: valid bits live in the high end of `acc`, unfilled low bits are
/// zero, which reproduces `BitReader`'s zero-padding-past-EOF semantics.
/// Keeping the next bits in a u64 avoids the per-read memory reload and bounds
/// machinery that `BitReader::read` incurs, so tight unpack loops stay in
/// registers. Reads must be <= 32 bits.
pub(crate) struct BitAcc<'a> {
    data: &'a [u8],
    byte_pos: usize,
    acc: u64,
    nbits: u32,
    /// Bits consumed relative to `base` (the starting bit position).
    consumed: usize,
    base: usize,
}

impl<'a> BitAcc<'a> {
    #[inline]
    fn new(data: &'a [u8], start: usize) -> Self {
        let mut s = Self {
            data,
            byte_pos: start >> 3,
            acc: 0,
            nbits: 0,
            consumed: 0,
            base: start,
        };
        s.refill();
        // Drop the already-consumed bits in the leading byte.
        let lead = (start & 7) as u32;
        s.acc <<= lead;
        s.nbits = s.nbits.saturating_sub(lead);
        s
    }

    #[inline]
    fn refill(&mut self) {
        while self.nbits <= 56 && self.byte_pos < self.data.len() {
            self.acc |= (self.data[self.byte_pos] as u64) << (56 - self.nbits);
            self.nbits += 8;
            self.byte_pos += 1;
        }
    }

    /// Read `bits` (1..=32) and advance. Reads past EOF yield zero bits.
    #[inline(always)]
    pub fn read(&mut self, bits: u32) -> u32 {
        debug_assert!((1..=32).contains(&bits));
        if self.nbits < 32 {
            self.refill();
        }
        let value = (self.acc >> (64 - bits)) as u32;
        self.acc <<= bits;
        self.nbits = self.nbits.saturating_sub(bits);
        self.consumed += bits as usize;
        value
    }

    /// Write the consumed bit count back to the reader this was forked from.
    #[inline]
    pub fn sync(&self, br: &mut BitReader<'_>) {
        br.set_position(self.base + self.consumed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bits() {
        let data = [0b10110100, 0b01101001];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read(1), 1); // 1
        assert_eq!(reader.read(1), 0); // 0
        assert_eq!(reader.read(2), 0b11); // 11
        assert_eq!(reader.read(4), 0b0100); // 0100
        assert_eq!(reader.read(4), 0b0110); // 0110
    }

    #[test]
    fn test_peek() {
        let data = [0b10110100];
        let reader = BitReader::new(&data);

        assert_eq!(reader.peek(4), 0b1011);
        assert_eq!(reader.peek(8), 0b10110100);
    }

    #[test]
    fn test_skip() {
        let data = [0b10110100, 0b01101001];
        let mut reader = BitReader::new(&data);

        reader.skip(4);
        assert_eq!(reader.read(4), 0b0100);
    }

    #[test]
    fn test_read_signed() {
        let data = [0b11110000];
        let mut reader = BitReader::new(&data);

        // 4 bits: 1111 = -1 when signed
        assert_eq!(reader.read_signed(4), -1);
    }

    #[test]
    fn test_bit_writer() {
        let mut writer = BitWriter::new(2);
        writer.write(0b1011, 4);
        writer.write(0b0100, 4);

        assert_eq!(writer.data()[0], 0b10110100);
    }

    #[test]
    fn test_multi_byte_read() {
        // Big-endian 0x12345678
        let data = [0x12, 0x34, 0x56, 0x78];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read(16), 0x1234);
        assert_eq!(reader.read(16), 0x5678);

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read(4), 0x1);
        assert_eq!(reader.read(4), 0x2);
        assert_eq!(reader.read(8), 0x34);
        assert_eq!(reader.read(12), 0x567);
    }

    #[test]
    fn test_eof_reads_return_zero() {
        // An out-of-range read returns a full zero (matches vgmstream clhca.c),
        // not the available high bits with zero-padded low bits.
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read(8), 0xFF);
        assert_eq!(reader.read(4), 0); // past EOF
        assert_eq!(reader.peek(16), 0);

        // A read that crosses the end returns 0 entirely, not a partial value.
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read(4), 0xF); // fully in range
        assert_eq!(reader.read(8), 0); // crosses EOF -> full zero
    }

    #[test]
    fn test_with_offset() {
        let data = [0xAB, 0xCD, 0xEF];
        let reader = BitReader::with_offset(&data, 1);
        assert_eq!(reader.position(), 8);
        assert_eq!(reader.peek(8), 0xCD);
    }

    #[test]
    fn test_reader_helpers_and_accumulator() {
        let data = [0b0100_1101, 0b0011_0101, 0xaa, 0x55, 0xf0];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.data(), &data);
        assert_eq!(reader.peek(0), 0);
        assert_eq!(reader.peek(33), 0);
        assert!(reader.has_bits(40));
        assert_eq!(reader.remaining_bits(), 40);

        assert_eq!(reader.read_hca_bits(0), 0);
        assert_eq!(reader.read_hca_bits(3), 0b010);
        reader.advance_signed(2);
        assert_eq!(reader.position(), 5);
        reader.advance_signed(-3);
        assert_eq!(reader.position(), 2);
        reader.advance_signed(-10);
        assert_eq!(reader.position(), 0);
        assert!(!reader.read_bit());
        assert_eq!(reader.remaining_bits(), 39);

        reader.set_position(3);
        let mut acc = reader.acc();
        assert_eq!(acc.read(5), 0b01101);
        assert_eq!(acc.read(8), 0b00110101);
        acc.sync(&mut reader);
        assert_eq!(reader.position(), 16);

        reader.set_position(35);
        assert_eq!(reader.read_hca_bits(4), 0b1000);
        assert_eq!(reader.remaining_bits(), 1);
        assert!(!reader.has_bits(2));
    }

    #[test]
    fn test_variable_values_and_writer_boundaries() {
        let mut encoded = BitWriter::new(3);
        encoded.write(0b0_1001_1010, 9);
        encoded.write(0, 0);
        encoded.write(u32::MAX, 33);
        assert_eq!(encoded.position(), 9);

        let mut reader = BitReader::new(encoded.data());
        assert_eq!(reader.read_scale_count(), (2, 10));

        let mut zero_scale = BitReader::new(&[0]);
        assert_eq!(zero_scale.read_scale_count(), (0, 0));
        assert_eq!(zero_scale.read_signed(0), 0);

        let mut positive = BitReader::new(&[0b0110_0000]);
        assert_eq!(positive.read_signed(4), 6);
        let mut off_by_one = BitReader::new(&[0b0100_0000]);
        assert_eq!(off_by_one.read_off_by_one(1), 0);
        assert_eq!(off_by_one.read_off_by_one(2), 3);

        let mut bounded = BitWriter::new(1);
        bounded.write(0xaa, 8);
        bounded.write(0xff, 8);
        assert_eq!(bounded.position(), 16);
        assert_eq!(bounded.into_vec(), vec![0xaa]);
    }
}
