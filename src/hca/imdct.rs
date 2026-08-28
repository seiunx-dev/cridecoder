//! IMDCT (Inverse Modified Discrete Cosine Transform) for HCA decoding

use super::decoder::{StChannel, HCA_SAMPLES_PER_SUBFRAME};
use super::tables::IMDCT_WINDOW;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_f32, vextq_f32, vld1q_f32, vld2q_f32, vmulq_f32, vrev64q_f32, vst1q_f32, vsubq_f32,
};

// x86_64 guarantees SSE2 in its baseline, so these are usable without a
// target_feature gate and run on every x86_64 CI target (Linux/macOS/Windows).
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permutevar8x32_ps, _mm256_set_epi32,
    _mm256_shuffle_ps, _mm256_storeu_ps, _mm256_sub_ps, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps,
    _mm_shuffle_ps, _mm_storeu_ps, _mm_sub_ps,
};

const HALF: usize = HCA_SAMPLES_PER_SUBFRAME / 2;

const fn table_from_bits(hex: [[u32; 64]; 7]) -> [[f32; 64]; 7] {
    let mut result = [[0.0f32; 64]; 7];
    let mut i = 0;
    while i < 7 {
        let mut j = 0;
        while j < 64 {
            result[i][j] = f32::from_bits(hex[i][j]);
            j += 1;
        }
        i += 1;
    }
    result
}

/// Sin tables for DCT-IV (7 x 64 arrays)
static SIN_TABLES: [[f32; 64]; 7] = table_from_bits([
    [
        0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75,
        0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75,
        0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75,
        0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75,
        0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75,
        0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75,
        0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75,
        0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75,
        0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75, 0x3DA73D75,
        0x3DA73D75,
    ],
    [
        0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE,
        0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31,
        0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE,
        0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31,
        0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE,
        0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31,
        0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE,
        0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31,
        0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE, 0x3F54DB31, 0x3F7B14BE,
        0x3F54DB31,
    ],
    [
        0x3F7EC46D, 0x3F74FA0B, 0x3F61C598, 0x3F45E403, 0x3F7EC46D, 0x3F74FA0B, 0x3F61C598,
        0x3F45E403, 0x3F7EC46D, 0x3F74FA0B, 0x3F61C598, 0x3F45E403, 0x3F7EC46D, 0x3F74FA0B,
        0x3F61C598, 0x3F45E403, 0x3F7EC46D, 0x3F74FA0B, 0x3F61C598, 0x3F45E403, 0x3F7EC46D,
        0x3F74FA0B, 0x3F61C598, 0x3F45E403, 0x3F7EC46D, 0x3F74FA0B, 0x3F61C598, 0x3F45E403,
        0x3F7EC46D, 0x3F74FA0B, 0x3F61C598, 0x3F45E403, 0x3F7EC46D, 0x3F74FA0B, 0x3F61C598,
        0x3F45E403, 0x3F7EC46D, 0x3F74FA0B, 0x3F61C598, 0x3F45E403, 0x3F7EC46D, 0x3F74FA0B,
        0x3F61C598, 0x3F45E403, 0x3F7EC46D, 0x3F74FA0B, 0x3F61C598, 0x3F45E403, 0x3F7EC46D,
        0x3F74FA0B, 0x3F61C598, 0x3F45E403, 0x3F7EC46D, 0x3F74FA0B, 0x3F61C598, 0x3F45E403,
        0x3F7EC46D, 0x3F74FA0B, 0x3F61C598, 0x3F45E403, 0x3F7EC46D, 0x3F74FA0B, 0x3F61C598,
        0x3F45E403,
    ],
    [
        0x3F7FB10F, 0x3F7D3AAC, 0x3F7853F8, 0x3F710908, 0x3F676BD8, 0x3F5B941A, 0x3F4D9F02,
        0x3F3DAEF9, 0x3F7FB10F, 0x3F7D3AAC, 0x3F7853F8, 0x3F710908, 0x3F676BD8, 0x3F5B941A,
        0x3F4D9F02, 0x3F3DAEF9, 0x3F7FB10F, 0x3F7D3AAC, 0x3F7853F8, 0x3F710908, 0x3F676BD8,
        0x3F5B941A, 0x3F4D9F02, 0x3F3DAEF9, 0x3F7FB10F, 0x3F7D3AAC, 0x3F7853F8, 0x3F710908,
        0x3F676BD8, 0x3F5B941A, 0x3F4D9F02, 0x3F3DAEF9, 0x3F7FB10F, 0x3F7D3AAC, 0x3F7853F8,
        0x3F710908, 0x3F676BD8, 0x3F5B941A, 0x3F4D9F02, 0x3F3DAEF9, 0x3F7FB10F, 0x3F7D3AAC,
        0x3F7853F8, 0x3F710908, 0x3F676BD8, 0x3F5B941A, 0x3F4D9F02, 0x3F3DAEF9, 0x3F7FB10F,
        0x3F7D3AAC, 0x3F7853F8, 0x3F710908, 0x3F676BD8, 0x3F5B941A, 0x3F4D9F02, 0x3F3DAEF9,
        0x3F7FB10F, 0x3F7D3AAC, 0x3F7853F8, 0x3F710908, 0x3F676BD8, 0x3F5B941A, 0x3F4D9F02,
        0x3F3DAEF9,
    ],
    [
        0x3F7FEC43, 0x3F7F4E6D, 0x3F7E1324, 0x3F7C3B28, 0x3F79C79D, 0x3F76BA07, 0x3F731447,
        0x3F6ED89E, 0x3F6A09A7, 0x3F64AA59, 0x3F5EBE05, 0x3F584853, 0x3F514D3D, 0x3F49D112,
        0x3F41D870, 0x3F396842, 0x3F7FEC43, 0x3F7F4E6D, 0x3F7E1324, 0x3F7C3B28, 0x3F79C79D,
        0x3F76BA07, 0x3F731447, 0x3F6ED89E, 0x3F6A09A7, 0x3F64AA59, 0x3F5EBE05, 0x3F584853,
        0x3F514D3D, 0x3F49D112, 0x3F41D870, 0x3F396842, 0x3F7FEC43, 0x3F7F4E6D, 0x3F7E1324,
        0x3F7C3B28, 0x3F79C79D, 0x3F76BA07, 0x3F731447, 0x3F6ED89E, 0x3F6A09A7, 0x3F64AA59,
        0x3F5EBE05, 0x3F584853, 0x3F514D3D, 0x3F49D112, 0x3F41D870, 0x3F396842, 0x3F7FEC43,
        0x3F7F4E6D, 0x3F7E1324, 0x3F7C3B28, 0x3F79C79D, 0x3F76BA07, 0x3F731447, 0x3F6ED89E,
        0x3F6A09A7, 0x3F64AA59, 0x3F5EBE05, 0x3F584853, 0x3F514D3D, 0x3F49D112, 0x3F41D870,
        0x3F396842,
    ],
    [
        0x3F7FFB11, 0x3F7FD397, 0x3F7F84AB, 0x3F7F0E58, 0x3F7E70B0, 0x3F7DABCC, 0x3F7CBFC9,
        0x3F7BACCD, 0x3F7A7302, 0x3F791298, 0x3F778BC5, 0x3F75DEC6, 0x3F740BDD, 0x3F721352,
        0x3F6FF573, 0x3F6DB293, 0x3F6B4B0C, 0x3F68BF3C, 0x3F660F88, 0x3F633C5A, 0x3F604621,
        0x3F5D2D53, 0x3F59F26A, 0x3F5695E5, 0x3F531849, 0x3F4F7A1F, 0x3F4BBBF8, 0x3F47DE65,
        0x3F43E200, 0x3F3FC767, 0x3F3B8F3B, 0x3F373A23, 0x3F7FFB11, 0x3F7FD397, 0x3F7F84AB,
        0x3F7F0E58, 0x3F7E70B0, 0x3F7DABCC, 0x3F7CBFC9, 0x3F7BACCD, 0x3F7A7302, 0x3F791298,
        0x3F778BC5, 0x3F75DEC6, 0x3F740BDD, 0x3F721352, 0x3F6FF573, 0x3F6DB293, 0x3F6B4B0C,
        0x3F68BF3C, 0x3F660F88, 0x3F633C5A, 0x3F604621, 0x3F5D2D53, 0x3F59F26A, 0x3F5695E5,
        0x3F531849, 0x3F4F7A1F, 0x3F4BBBF8, 0x3F47DE65, 0x3F43E200, 0x3F3FC767, 0x3F3B8F3B,
        0x3F373A23,
    ],
    [
        0x3F7FFEC4, 0x3F7FF4E6, 0x3F7FE129, 0x3F7FC38F, 0x3F7F9C18, 0x3F7F6AC7, 0x3F7F2F9D,
        0x3F7EEA9D, 0x3F7E9BC9, 0x3F7E4323, 0x3F7DE0B1, 0x3F7D7474, 0x3F7CFE73, 0x3F7C7EB0,
        0x3F7BF531, 0x3F7B61FC, 0x3F7AC516, 0x3F7A1E84, 0x3F796E4E, 0x3F78B47B, 0x3F77F110,
        0x3F772417, 0x3F764D97, 0x3F756D97, 0x3F748422, 0x3F73913F, 0x3F7294F8, 0x3F718F57,
        0x3F708066, 0x3F6F6830, 0x3F6E46BE, 0x3F6D1C1D, 0x3F6BE858, 0x3F6AAB7B, 0x3F696591,
        0x3F6816A8, 0x3F66BECC, 0x3F655E0B, 0x3F63F473, 0x3F628210, 0x3F6106F2, 0x3F5F8327,
        0x3F5DF6BE, 0x3F5C61C7, 0x3F5AC450, 0x3F591E6A, 0x3F577026, 0x3F55B993, 0x3F53FAC3,
        0x3F5233C6, 0x3F5064AF, 0x3F4E8D90, 0x3F4CAE79, 0x3F4AC77F, 0x3F48D8B3, 0x3F46E22A,
        0x3F44E3F5, 0x3F42DE29, 0x3F40D0DA, 0x3F3EBC1B, 0x3F3CA003, 0x3F3A7CA4, 0x3F385216,
        0x3F36206C,
    ],
]);

/// Cos tables for DCT-IV (7 x 64 arrays)
static COS_TABLES: [[f32; 64]; 7] = table_from_bits([
    [
        0xBD0A8BD4, 0x3D0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0xBD0A8BD4,
        0x3D0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4,
        0x3D0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4,
        0x3D0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4,
        0x3D0A8BD4, 0xBD0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0xBD0A8BD4,
        0x3D0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4,
        0x3D0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4,
        0x3D0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4,
        0x3D0A8BD4, 0xBD0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0xBD0A8BD4, 0x3D0A8BD4, 0x3D0A8BD4,
        0xBD0A8BD4,
    ],
    [
        0xBE47C5C2, 0xBF0E39DA, 0x3E47C5C2, 0x3F0E39DA, 0x3E47C5C2, 0x3F0E39DA, 0xBE47C5C2,
        0xBF0E39DA, 0x3E47C5C2, 0x3F0E39DA, 0xBE47C5C2, 0xBF0E39DA, 0xBE47C5C2, 0xBF0E39DA,
        0x3E47C5C2, 0x3F0E39DA, 0x3E47C5C2, 0x3F0E39DA, 0xBE47C5C2, 0xBF0E39DA, 0xBE47C5C2,
        0xBF0E39DA, 0x3E47C5C2, 0x3F0E39DA, 0xBE47C5C2, 0xBF0E39DA, 0x3E47C5C2, 0x3F0E39DA,
        0x3E47C5C2, 0x3F0E39DA, 0xBE47C5C2, 0xBF0E39DA, 0x3E47C5C2, 0x3F0E39DA, 0xBE47C5C2,
        0xBF0E39DA, 0xBE47C5C2, 0xBF0E39DA, 0x3E47C5C2, 0x3F0E39DA, 0xBE47C5C2, 0xBF0E39DA,
        0x3E47C5C2, 0x3F0E39DA, 0x3E47C5C2, 0x3F0E39DA, 0xBE47C5C2, 0xBF0E39DA, 0xBE47C5C2,
        0xBF0E39DA, 0x3E47C5C2, 0x3F0E39DA, 0x3E47C5C2, 0x3F0E39DA, 0xBE47C5C2, 0xBF0E39DA,
        0x3E47C5C2, 0x3F0E39DA, 0xBE47C5C2, 0xBF0E39DA, 0xBE47C5C2, 0xBF0E39DA, 0x3E47C5C2,
        0x3F0E39DA,
    ],
    [
        0xBDC8BD36, 0xBE94A031, 0xBEF15AEA, 0xBF226799, 0x3DC8BD36, 0x3E94A031, 0x3EF15AEA,
        0x3F226799, 0x3DC8BD36, 0x3E94A031, 0x3EF15AEA, 0x3F226799, 0xBDC8BD36, 0xBE94A031,
        0xBEF15AEA, 0xBF226799, 0x3DC8BD36, 0x3E94A031, 0x3EF15AEA, 0x3F226799, 0xBDC8BD36,
        0xBE94A031, 0xBEF15AEA, 0xBF226799, 0xBDC8BD36, 0xBE94A031, 0xBEF15AEA, 0xBF226799,
        0x3DC8BD36, 0x3E94A031, 0x3EF15AEA, 0x3F226799, 0x3DC8BD36, 0x3E94A031, 0x3EF15AEA,
        0x3F226799, 0xBDC8BD36, 0xBE94A031, 0xBEF15AEA, 0xBF226799, 0xBDC8BD36, 0xBE94A031,
        0xBEF15AEA, 0xBF226799, 0x3DC8BD36, 0x3E94A031, 0x3EF15AEA, 0x3F226799, 0xBDC8BD36,
        0xBE94A031, 0xBEF15AEA, 0xBF226799, 0x3DC8BD36, 0x3E94A031, 0x3EF15AEA, 0x3F226799,
        0x3DC8BD36, 0x3E94A031, 0x3EF15AEA, 0x3F226799, 0xBDC8BD36, 0xBE94A031, 0xBEF15AEA,
        0xBF226799,
    ],
    [
        0xBD48FB30, 0xBE164083, 0xBE78CFCC, 0xBEAC7CD4, 0xBEDAE880, 0xBF039C3D, 0xBF187FC0,
        0xBF2BEB4A, 0x3D48FB30, 0x3E164083, 0x3E78CFCC, 0x3EAC7CD4, 0x3EDAE880, 0x3F039C3D,
        0x3F187FC0, 0x3F2BEB4A, 0x3D48FB30, 0x3E164083, 0x3E78CFCC, 0x3EAC7CD4, 0x3EDAE880,
        0x3F039C3D, 0x3F187FC0, 0x3F2BEB4A, 0xBD48FB30, 0xBE164083, 0xBE78CFCC, 0xBEAC7CD4,
        0xBEDAE880, 0xBF039C3D, 0xBF187FC0, 0xBF2BEB4A, 0x3D48FB30, 0x3E164083, 0x3E78CFCC,
        0x3EAC7CD4, 0x3EDAE880, 0x3F039C3D, 0x3F187FC0, 0x3F2BEB4A, 0xBD48FB30, 0xBE164083,
        0xBE78CFCC, 0xBEAC7CD4, 0xBEDAE880, 0xBF039C3D, 0xBF187FC0, 0xBF2BEB4A, 0xBD48FB30,
        0xBE164083, 0xBE78CFCC, 0xBEAC7CD4, 0xBEDAE880, 0xBF039C3D, 0xBF187FC0, 0xBF2BEB4A,
        0x3D48FB30, 0x3E164083, 0x3E78CFCC, 0x3EAC7CD4, 0x3EDAE880, 0x3F039C3D, 0x3F187FC0,
        0x3F2BEB4A,
    ],
    [
        0xBCC90AB0, 0xBD96A905, 0xBDFAB273, 0xBE2F10A2, 0xBE605C13, 0xBE888E93, 0xBEA09AE5,
        0xBEB8442A, 0xBECF7BCA, 0xBEE63375, 0xBEFC5D27, 0xBF08F59B, 0xBF13682A, 0xBF1D7FD1,
        0xBF273656, 0xBF3085BB, 0x3CC90AB0, 0x3D96A905, 0x3DFAB273, 0x3E2F10A2, 0x3E605C13,
        0x3E888E93, 0x3EA09AE5, 0x3EB8442A, 0x3ECF7BCA, 0x3EE63375, 0x3EFC5D27, 0x3F08F59B,
        0x3F13682A, 0x3F1D7FD1, 0x3F273656, 0x3F3085BB, 0x3CC90AB0, 0x3D96A905, 0x3DFAB273,
        0x3E2F10A2, 0x3E605C13, 0x3E888E93, 0x3EA09AE5, 0x3EB8442A, 0x3ECF7BCA, 0x3EE63375,
        0x3EFC5D27, 0x3F08F59B, 0x3F13682A, 0x3F1D7FD1, 0x3F273656, 0x3F3085BB, 0xBCC90AB0,
        0xBD96A905, 0xBDFAB273, 0xBE2F10A2, 0xBE605C13, 0xBE888E93, 0xBEA09AE5, 0xBEB8442A,
        0xBECF7BCA, 0xBEE63375, 0xBEFC5D27, 0xBF08F59B, 0xBF13682A, 0xBF1D7FD1, 0xBF273656,
        0xBF3085BB,
    ],
    [
        0xBC490E90, 0xBD16C32C, 0xBD7B2B74, 0xBDAFB680, 0xBDE1BC2E, 0xBE09CF86, 0xBE22ABB6,
        0xBE3B6ECF, 0xBE541501, 0xBE6C9A7F, 0xBE827DC0, 0xBE8E9A22, 0xBE9AA086, 0xBEA68F12,
        0xBEB263EF, 0xBEBE1D4A, 0xBEC9B953, 0xBED53641, 0xBEE0924F, 0xBEEBCBBB, 0xBEF6E0CB,
        0xBF00E7E4, 0xBF064B82, 0xBF0B9A6B, 0xBF10D3CD, 0xBF15F6D9, 0xBF1B02C6, 0xBF1FF6CB,
        0xBF24D225, 0xBF299415, 0xBF2E3BDE, 0xBF32C8C9, 0x3C490E90, 0x3D16C32C, 0x3D7B2B74,
        0x3DAFB680, 0x3DE1BC2E, 0x3E09CF86, 0x3E22ABB6, 0x3E3B6ECF, 0x3E541501, 0x3E6C9A7F,
        0x3E827DC0, 0x3E8E9A22, 0x3E9AA086, 0x3EA68F12, 0x3EB263EF, 0x3EBE1D4A, 0x3EC9B953,
        0x3ED53641, 0x3EE0924F, 0x3EEBCBBB, 0x3EF6E0CB, 0x3F00E7E4, 0x3F064B82, 0x3F0B9A6B,
        0x3F10D3CD, 0x3F15F6D9, 0x3F1B02C6, 0x3F1FF6CB, 0x3F24D225, 0x3F299415, 0x3F2E3BDE,
        0x3F32C8C9,
    ],
    [
        0xBBC90F88, 0xBC96C9B6, 0xBCFB49BA, 0xBD2FE007, 0xBD621469, 0xBD8A200A, 0xBDA3308C,
        0xBDBC3AC3, 0xBDD53DB9, 0xBDEE3876, 0xBE039502, 0xBE1008B7, 0xBE1C76DE, 0xBE28DEFC,
        0xBE354098, 0xBE419B37, 0xBE4DEE60, 0xBE5A3997, 0xBE667C66, 0xBE72B651, 0xBE7EE6E1,
        0xBE8586CE, 0xBE8B9507, 0xBE919DDD, 0xBE97A117, 0xBE9D9E78, 0xBEA395C5, 0xBEA986C4,
        0xBEAF713A, 0xBEB554EC, 0xBEBB31A0, 0xBEC1071E, 0xBEC6D529, 0xBECC9B8B, 0xBED25A09,
        0xBED8106B, 0xBEDDBE79, 0xBEE363FA, 0xBEE900B7, 0xBEEE9479, 0xBEF41F07, 0xBEF9A02D,
        0xBEFF17B2, 0xBF0242B1, 0xBF04F484, 0xBF07A136, 0xBF0A48AD, 0xBF0CEAD0, 0xBF0F8784,
        0xBF121EB0, 0xBF14B039, 0xBF173C07, 0xBF19C200, 0xBF1C420C, 0xBF1EBC12, 0xBF212FF9,
        0xBF239DA9, 0xBF26050A, 0xBF286605, 0xBF2AC082, 0xBF2D1469, 0xBF2F61A5, 0xBF31A81D,
        0xBF33E7BC,
    ],
]);

/// Apply IMDCT transform to dequantized spectra.
///
/// On x86_64 the AVX2-vs-SSE2 choice is made ONCE here (whole-transform
/// multiversioning) so the per-stage SIMD helpers fully inline into one
/// function — per-stage runtime dispatch would force non-inlinable
/// `#[target_feature]` calls and lose the win. Other arches ignore AVX2 and
/// use NEON/scalar via the stage dispatch.
pub fn imdct_transform(ch: &mut StChannel, subframe: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { imdct_transform_avx2(ch, subframe) }
        } else {
            imdct_body::<false>(ch, subframe)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    imdct_body::<false>(ch, subframe);
}

// AVX2 whole-transform entry: target_feature so imdct_body<true> and its AVX2
// stage helpers all inline into this single function. Reached only after the
// runtime is_x86_feature_detected!("avx2") check above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn imdct_transform_avx2(ch: &mut StChannel, subframe: usize) {
    imdct_body::<true>(ch, subframe);
}

/// DCT stages only (no overlap-add), leaving the DCT-IV result in
/// `ch.spectra[subframe]`. Used by the block-parallel decode path, where the
/// serial overlap-add runs later via `imdct_overlap`.
pub(crate) fn imdct_dct(ch: &mut StChannel, subframe: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { imdct_dct_avx2(ch, subframe) }
        } else {
            dct_stages::<false>(ch.spectra[subframe].as_mut_ptr(), ch.temp.as_mut_ptr())
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    dct_stages::<false>(ch.spectra[subframe].as_mut_ptr(), ch.temp.as_mut_ptr());
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn imdct_dct_avx2(ch: &mut StChannel, subframe: usize) {
    dct_stages::<true>(ch.spectra[subframe].as_mut_ptr(), ch.temp.as_mut_ptr());
}

/// Overlap-add window on standalone 128-sample slices, bit-identical to the
/// tail of `imdct_body`. `dct` is a DCT-IV output (from `imdct_dct`),
/// `previous` carries the running overlap state, `wave` receives the samples.
pub(crate) fn imdct_overlap(
    dct: &[f32; HCA_SAMPLES_PER_SUBFRAME],
    previous: &mut [f32; HCA_SAMPLES_PER_SUBFRAME],
    wave: &mut [f32; HCA_SAMPLES_PER_SUBFRAME],
) {
    let window = &IMDCT_WINDOW;
    for i in 0..HALF {
        wave[i] = window[i] * dct[i + HALF] + previous[i];
        wave[i + HALF] =
            window[i + HALF] * dct[HCA_SAMPLES_PER_SUBFRAME - 1 - i] - previous[i + HALF];
    }
    for i in 0..HALF {
        previous[i] = window[HCA_SAMPLES_PER_SUBFRAME - 1 - i] * dct[HALF - i - 1];
        previous[i + HALF] = window[HALF - i - 1] * dct[i];
    }
}

#[inline(always)]
fn dct_stages<const AVX2: bool>(spectra: *mut f32, temp: *mut f32) {
    butterfly_stage::<1, 64, false, AVX2>(spectra, temp);
    butterfly_stage::<2, 32, true, AVX2>(spectra, temp);
    butterfly_stage::<4, 16, false, AVX2>(spectra, temp);
    butterfly_stage::<8, 8, true, AVX2>(spectra, temp);
    butterfly_stage::<16, 4, false, AVX2>(spectra, temp);
    butterfly_stage::<32, 2, true, AVX2>(spectra, temp);
    butterfly_stage::<64, 1, false, AVX2>(spectra, temp);

    dct_stage::<0, 64, 1, true, AVX2>(spectra, temp);
    dct_stage::<1, 32, 2, false, AVX2>(spectra, temp);
    dct_stage::<2, 16, 4, true, AVX2>(spectra, temp);
    dct_stage::<3, 8, 8, false, AVX2>(spectra, temp);
    dct_stage::<4, 4, 16, true, AVX2>(spectra, temp);
    dct_stage::<5, 2, 32, false, AVX2>(spectra, temp);
    dct_stage::<6, 1, 64, true, AVX2>(spectra, temp);
}

#[inline(always)]
fn imdct_body<const AVX2: bool>(ch: &mut StChannel, subframe: usize) {
    // The IMDCT operates on fixed 128-sample buffers. The stage schedule
    // only indexes within those arrays.
    let spectra = ch.spectra[subframe].as_mut_ptr();
    let temp = ch.temp.as_mut_ptr();

    dct_stages::<AVX2>(spectra, temp);

    // Update output/IMDCT with overlapped window
    let wave = ch.wave[subframe].as_mut_ptr();
    let previous = ch.imdct_previous.as_mut_ptr();
    let window = IMDCT_WINDOW.as_ptr();
    for i in 0..HALF {
        unsafe {
            *wave.add(i) = *window.add(i) * *spectra.add(i + HALF) + *previous.add(i);
            *wave.add(i + HALF) = *window.add(i + HALF)
                * *spectra.add(HCA_SAMPLES_PER_SUBFRAME - 1 - i)
                - *previous.add(i + HALF);
            *previous.add(i) =
                *window.add(HCA_SAMPLES_PER_SUBFRAME - 1 - i) * *spectra.add(HALF - i - 1);
            *previous.add(i + HALF) = *window.add(HALF - i - 1) * *spectra.add(i);
        }
    }
}

#[inline(always)]
fn butterfly_stage<
    const COUNT1: usize,
    const COUNT2: usize,
    const TEMP_SRC: bool,
    const AVX2: bool,
>(
    spectra: *mut f32,
    temp: *mut f32,
) {
    // TEMP_SRC: read temp, write spectra; else read spectra, write temp.
    let (src, dst): (*const f32, *mut f32) = if TEMP_SRC {
        (temp, spectra)
    } else {
        (spectra, temp)
    };

    #[cfg(target_arch = "aarch64")]
    if COUNT2 >= 4 {
        unsafe { butterfly_neon::<COUNT1, COUNT2>(src, dst) };
        return;
    }

    // AVX2 vs SSE2 chosen at compile time via the AVX2 const (set once by the
    // whole-transform dispatcher), so the chosen helper inlines.
    #[cfg(target_arch = "x86_64")]
    {
        if AVX2 && COUNT2 >= 8 {
            unsafe { butterfly_avx2::<COUNT1, COUNT2>(src, dst) };
            return;
        }
        if COUNT2 >= 4 {
            unsafe { butterfly_sse::<COUNT1, COUNT2>(src, dst) };
            return;
        }
    }

    // Scalar fallback (COUNT2 < 4, or no SIMD path). Bit-identical to the SIMD path.
    let mut d1_idx = 0usize;
    let mut d2_idx = COUNT2;
    let mut t1_idx = 0usize;
    for _ in 0..COUNT1 {
        for _ in 0..COUNT2 {
            unsafe {
                let a = *src.add(t1_idx);
                let b = *src.add(t1_idx + 1);
                t1_idx += 2;
                *dst.add(d1_idx) = a + b;
                *dst.add(d2_idx) = a - b;
            }
            d1_idx += 1;
            d2_idx += 1;
        }
        d1_idx += COUNT2;
        d2_idx += COUNT2;
    }
}

// NEON butterfly: deinterleave a/b pairs (vld2q), write a+b ascending and a-b
// ascending. COUNT2 is always a multiple of 4 here. Bit-identical to scalar
// (same IEEE add/sub, no FMA).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn butterfly_neon<const COUNT1: usize, const COUNT2: usize>(src: *const f32, dst: *mut f32) {
    let mut d1 = 0usize;
    let mut d2 = COUNT2;
    let mut t = 0usize;
    for _ in 0..COUNT1 {
        let mut k = 0;
        while k + 4 <= COUNT2 {
            let ab = vld2q_f32(src.add(t)); // ab.0 = a[4], ab.1 = b[4]
            vst1q_f32(dst.add(d1), vaddq_f32(ab.0, ab.1));
            vst1q_f32(dst.add(d2), vsubq_f32(ab.0, ab.1));
            t += 8;
            d1 += 4;
            d2 += 4;
            k += 4;
        }
        d1 += COUNT2;
        d2 += COUNT2;
    }
}

// SSE2 butterfly (x86_64 baseline), 4-wide. Deinterleave a/b pairs via
// _mm_shuffle_ps, write a+b and a-b. Bit-identical to scalar (IEEE add/sub, no FMA).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn butterfly_sse<const COUNT1: usize, const COUNT2: usize>(src: *const f32, dst: *mut f32) {
    let mut d1 = 0usize;
    let mut d2 = COUNT2;
    let mut t = 0usize;
    for _ in 0..COUNT1 {
        let mut k = 0;
        while k + 4 <= COUNT2 {
            let v0 = _mm_loadu_ps(src.add(t)); // [a0,b0,a1,b1]
            let v1 = _mm_loadu_ps(src.add(t + 4)); // [a2,b2,a3,b3]
            let a = _mm_shuffle_ps::<0x88>(v0, v1); // [a0,a1,a2,a3]
            let b = _mm_shuffle_ps::<0xDD>(v0, v1); // [b0,b1,b2,b3]
            _mm_storeu_ps(dst.add(d1), _mm_add_ps(a, b));
            _mm_storeu_ps(dst.add(d2), _mm_sub_ps(a, b));
            t += 8;
            d1 += 4;
            d2 += 4;
            k += 4;
        }
        d1 += COUNT2;
        d2 += COUNT2;
    }
}

// AVX2 butterfly (8-wide). Per-lane shuffle leaves [a0,a1,a4,a5,a2,a3,a6,a7];
// a cross-lane permutevar8x32 restores contiguous order, then a+b / a-b.
// COUNT2>=8. Bit-identical to scalar (IEEE add/sub, no FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn butterfly_avx2<const COUNT1: usize, const COUNT2: usize>(src: *const f32, dst: *mut f32) {
    let deint = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    let mut d1 = 0usize;
    let mut d2 = COUNT2;
    let mut t = 0usize;
    for _ in 0..COUNT1 {
        let mut k = 0;
        while k + 8 <= COUNT2 {
            let v0 = _mm256_loadu_ps(src.add(t));
            let v1 = _mm256_loadu_ps(src.add(t + 8));
            let a = _mm256_permutevar8x32_ps(_mm256_shuffle_ps::<0x88>(v0, v1), deint);
            let b = _mm256_permutevar8x32_ps(_mm256_shuffle_ps::<0xDD>(v0, v1), deint);
            _mm256_storeu_ps(dst.add(d1), _mm256_add_ps(a, b));
            _mm256_storeu_ps(dst.add(d2), _mm256_sub_ps(a, b));
            t += 16;
            d1 += 8;
            d2 += 8;
            k += 8;
        }
        d1 += COUNT2;
        d2 += COUNT2;
    }
}

#[inline(always)]
fn dct_stage<
    const TABLE: usize,
    const COUNT1: usize,
    const COUNT2: usize,
    const TEMP_SRC: bool,
    const AVX2: bool,
>(
    spectra: *mut f32,
    temp: *mut f32,
) {
    let (src, dst): (*const f32, *mut f32) = if TEMP_SRC {
        (temp, spectra)
    } else {
        (spectra, temp)
    };
    let sin_table = &SIN_TABLES[TABLE];
    let cos_table = &COS_TABLES[TABLE];

    #[cfg(target_arch = "aarch64")]
    if COUNT2 >= 4 {
        unsafe { dct_neon::<COUNT1, COUNT2>(src, dst, sin_table.as_ptr(), cos_table.as_ptr()) };
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if AVX2 && COUNT2 >= 8 {
            unsafe { dct_avx2::<COUNT1, COUNT2>(src, dst, sin_table.as_ptr(), cos_table.as_ptr()) };
            return;
        }
        if COUNT2 >= 4 {
            unsafe { dct_sse::<COUNT1, COUNT2>(src, dst, sin_table.as_ptr(), cos_table.as_ptr()) };
            return;
        }
    }

    // Scalar fallback (COUNT2 < 4, or no SIMD path). Bit-identical to the SIMD path.
    let mut sin_idx = 0;
    let mut cos_idx = 0;
    let mut d1_idx = 0usize;
    let mut d2_idx = COUNT2 * 2 - 1;
    let mut s1_idx = 0usize;
    let mut s2_idx = COUNT2;
    for _ in 0..COUNT1 {
        for _ in 0..COUNT2 {
            unsafe {
                let a = *src.add(s1_idx);
                let b = *src.add(s2_idx);
                s1_idx += 1;
                s2_idx += 1;
                let sin = sin_table[sin_idx];
                let cos = cos_table[cos_idx];
                sin_idx += 1;
                cos_idx += 1;
                *dst.add(d1_idx) = a * sin - b * cos;
                *dst.add(d2_idx) = a * cos + b * sin;
            }
            d1_idx += 1;
            d2_idx -= 1;
        }
        s1_idx += COUNT2;
        s2_idx += COUNT2;
        d1_idx += COUNT2;
        d2_idx += COUNT2 * 3;
    }
}

// NEON DCT-IV butterfly. d1 ascending = a*sin - b*cos; d2 descends (d2, d2-1,
// d2-2, d2-3) = a*cos + b*sin, so compute the vector, lane-reverse it, and store
// at the block's low end (d2-3). COUNT2 is a multiple of 4 here. Uses
// vmulq + vsubq/vaddq (no FMA) to stay bit-identical to the scalar path.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dct_neon<const COUNT1: usize, const COUNT2: usize>(
    src: *const f32,
    dst: *mut f32,
    sin_table: *const f32,
    cos_table: *const f32,
) {
    let mut tbl = 0usize;
    let mut d1 = 0usize;
    let mut d2 = COUNT2 * 2 - 1;
    let mut s1 = 0usize;
    let mut s2 = COUNT2;
    for _ in 0..COUNT1 {
        let mut k = 0;
        while k + 4 <= COUNT2 {
            let a = vld1q_f32(src.add(s1));
            let b = vld1q_f32(src.add(s2));
            let sin = vld1q_f32(sin_table.add(tbl));
            let cos = vld1q_f32(cos_table.add(tbl));
            vst1q_f32(dst.add(d1), vsubq_f32(vmulq_f32(a, sin), vmulq_f32(b, cos)));
            let d2v = vaddq_f32(vmulq_f32(a, cos), vmulq_f32(b, sin));
            let rev = vextq_f32::<2>(vrev64q_f32(d2v), vrev64q_f32(d2v));
            vst1q_f32(dst.add(d2 - 3), rev);
            s1 += 4;
            s2 += 4;
            tbl += 4;
            d1 += 4;
            d2 -= 4;
            k += 4;
        }
        s1 += COUNT2;
        s2 += COUNT2;
        d1 += COUNT2;
        d2 += COUNT2 * 3;
    }
}

// SSE2 DCT-IV butterfly (x86_64 baseline). d1 ascending = a*sin - b*cos; d2
// descends, so compute a*cos + b*sin, reverse the 4 lanes (_mm_shuffle_ps 0x1B),
// and store at the block's low end (d2-3). Bit-identical to scalar (no FMA).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn dct_sse<const COUNT1: usize, const COUNT2: usize>(
    src: *const f32,
    dst: *mut f32,
    sin_table: *const f32,
    cos_table: *const f32,
) {
    let mut tbl = 0usize;
    let mut d1 = 0usize;
    let mut d2 = COUNT2 * 2 - 1;
    let mut s1 = 0usize;
    let mut s2 = COUNT2;
    for _ in 0..COUNT1 {
        let mut k = 0;
        while k + 4 <= COUNT2 {
            let a = _mm_loadu_ps(src.add(s1));
            let b = _mm_loadu_ps(src.add(s2));
            let sin = _mm_loadu_ps(sin_table.add(tbl));
            let cos = _mm_loadu_ps(cos_table.add(tbl));
            _mm_storeu_ps(
                dst.add(d1),
                _mm_sub_ps(_mm_mul_ps(a, sin), _mm_mul_ps(b, cos)),
            );
            let d2v = _mm_add_ps(_mm_mul_ps(a, cos), _mm_mul_ps(b, sin));
            let rev = _mm_shuffle_ps::<0x1B>(d2v, d2v); // [v3,v2,v1,v0]
            _mm_storeu_ps(dst.add(d2 - 3), rev);
            s1 += 4;
            s2 += 4;
            tbl += 4;
            d1 += 4;
            d2 -= 4;
            k += 4;
        }
        s1 += COUNT2;
        s2 += COUNT2;
        d1 += COUNT2;
        d2 += COUNT2 * 3;
    }
}

// AVX2 DCT-IV butterfly (8-wide). d1 = a*sin - b*cos ascending; d2 = a*cos +
// b*sin reversed across all 8 lanes (permutevar8x32) and stored at d2-7.
// COUNT2>=8. No FMA -> bit-identical to scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dct_avx2<const COUNT1: usize, const COUNT2: usize>(
    src: *const f32,
    dst: *mut f32,
    sin_table: *const f32,
    cos_table: *const f32,
) {
    let rev = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut tbl = 0usize;
    let mut d1 = 0usize;
    let mut d2 = COUNT2 * 2 - 1;
    let mut s1 = 0usize;
    let mut s2 = COUNT2;
    for _ in 0..COUNT1 {
        let mut k = 0;
        while k + 8 <= COUNT2 {
            let a = _mm256_loadu_ps(src.add(s1));
            let b = _mm256_loadu_ps(src.add(s2));
            let sin = _mm256_loadu_ps(sin_table.add(tbl));
            let cos = _mm256_loadu_ps(cos_table.add(tbl));
            _mm256_storeu_ps(
                dst.add(d1),
                _mm256_sub_ps(_mm256_mul_ps(a, sin), _mm256_mul_ps(b, cos)),
            );
            let d2v = _mm256_add_ps(_mm256_mul_ps(a, cos), _mm256_mul_ps(b, sin));
            _mm256_storeu_ps(dst.add(d2 - 7), _mm256_permutevar8x32_ps(d2v, rev));
            s1 += 8;
            s2 += 8;
            tbl += 8;
            d1 += 8;
            d2 -= 8;
            k += 8;
        }
        s1 += COUNT2;
        s2 += COUNT2;
        d1 += COUNT2;
        d2 += COUNT2 * 3;
    }
}
