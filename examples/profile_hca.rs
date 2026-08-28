//! HCA decode profiling harness. Decodes the file many times in a tight loop so
//! a sampling profiler (`sample <pid>`) can attribute CPU time to hot functions.
//!
//! Also reports pure-Rust decode timing (to /dev/null, no WAV Vec growth and no
//! input copy), isolating the decoder core from Python-binding overhead.

use cridecoder::HcaDecoder;
use std::env;
use std::io::sink;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let hca_file = args.get(1).map(String::as_str).unwrap_or("music_5031.hca");
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);

    let probe = HcaDecoder::from_file(hca_file).expect("open HCA");
    println!(
        "profiling: {} iters, {} blocks, {} ch, sr {}",
        iters,
        probe.info().block_count,
        probe.info().channel_count,
        probe.info().sampling_rate
    );

    // Warm up once.
    {
        let mut d = HcaDecoder::from_file(hca_file).unwrap();
        d.decode_to_wav(&mut sink()).unwrap();
    }

    // Timed pure-decode loop: decode to /dev/null so we measure only the core.
    let start = Instant::now();
    let mut total = 0u64;
    for _ in 0..iters {
        let mut d = HcaDecoder::from_file(hca_file).unwrap();
        d.decode_to_wav(&mut sink()).unwrap();
        total += 1;
    }
    let elapsed = start.elapsed();
    println!(
        "pure decode core: {} iters in {:?} = {:.3} ms/decode (to /dev/null)",
        iters,
        elapsed,
        elapsed.as_secs_f64() * 1000.0 / iters as f64
    );
    std::hint::black_box(total);
}
