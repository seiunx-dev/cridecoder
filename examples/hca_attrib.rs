//! In-memory HCA decode benchmark: decodes every block from a preloaded buffer
//! (no file I/O in the loop) and prints an FNV hash of the PCM output so
//! optimizations can be checked for bit-exactness against a baseline.
use cridecoder::hca::ClHca;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let hca_file = args.get(1).map(String::as_str).unwrap_or("music_5031.hca");
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);

    let data = std::fs::read(hca_file).unwrap();
    let mut hca = Box::new(ClHca::new());
    hca.decode_header(&data).unwrap();
    let info = hca.get_info().unwrap();
    println!(
        "blocks={} block_size={} ch={} encrypted={}",
        info.block_count, info.block_size, info.channel_count, info.encryption_enabled
    );
    println!(
        "version={:#x} min_res={} max_res={} stereo_bands={} hfr_bands_per_group={} ms_stereo={}",
        hca.version,
        hca.min_resolution,
        hca.max_resolution,
        hca.stereo_band_count,
        hca.bands_per_hfr_group,
        hca.ms_stereo
    );

    let hs = info.header_size as usize;
    let bs = info.block_size as usize;
    let n = info.block_count as usize;
    let mut frame = vec![0u8; bs];
    let mut pcm = vec![0i16; info.samples_per_block * info.channel_count as usize];

    // Correctness hash over one full decode.
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in 0..n {
        frame.copy_from_slice(&data[hs + b * bs..hs + (b + 1) * bs]);
        hca.decode_block(&mut frame).unwrap();
        hca.read_samples_16(&mut pcm);
        for &s in &pcm {
            hash = (hash ^ (s as u16 as u64)).wrapping_mul(0x100000001b3);
        }
    }
    println!("pcm fnv hash: {:016x}", hash);

    let start = Instant::now();
    for _ in 0..iters {
        for b in 0..n {
            frame.copy_from_slice(&data[hs + b * bs..hs + (b + 1) * bs]);
            hca.decode_block(&mut frame).unwrap();
            hca.read_samples_16(&mut pcm);
            std::hint::black_box(&pcm);
        }
    }
    let el = start.elapsed();
    println!(
        "in-memory decode: {:.3} ms/decode",
        el.as_secs_f64() * 1000.0 / iters as f64
    );
}
