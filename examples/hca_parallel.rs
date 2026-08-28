//! Benchmark and verify decode_to_wav_parallel against the serial decoder.
use cridecoder::HcaDecoder;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let hca_file = args.get(1).map(String::as_str).unwrap_or("music_5031.hca");
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
    });

    // Correctness: byte-identical WAV output.
    let mut serial_out = Vec::new();
    let mut parallel_out = Vec::new();
    {
        let mut d = HcaDecoder::from_file(hca_file).unwrap();
        d.decode_to_wav(&mut serial_out).unwrap();
        let mut d = HcaDecoder::from_file(hca_file).unwrap();
        d.decode_to_wav_parallel(&mut parallel_out, threads)
            .unwrap();
    }
    assert_eq!(serial_out.len(), parallel_out.len(), "length mismatch");
    assert!(serial_out == parallel_out, "WAV bytes differ");
    println!(
        "byte-identical: yes ({} bytes, {} threads)",
        serial_out.len(),
        threads
    );

    let start = Instant::now();
    for _ in 0..iters {
        let mut d = HcaDecoder::from_file(hca_file).unwrap();
        let mut sink = std::io::sink();
        d.decode_to_wav(&mut sink).unwrap();
    }
    let serial = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        let mut d = HcaDecoder::from_file(hca_file).unwrap();
        let mut sink = std::io::sink();
        d.decode_to_wav_parallel(&mut sink, threads).unwrap();
    }
    let parallel = start.elapsed();

    println!(
        "serial:   {:.3} ms/decode",
        serial.as_secs_f64() * 1000.0 / iters as f64
    );
    println!(
        "parallel: {:.3} ms/decode ({} threads, {:.2}x)",
        parallel.as_secs_f64() * 1000.0 / iters as f64,
        threads,
        serial.as_secs_f64() / parallel.as_secs_f64()
    );
}
