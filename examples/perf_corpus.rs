//! Production-shaped throughput harness and output digests for a local corpus.
//!
//! Mirrors how Haruki-Sekai-Asset-Updater drives cridecoder:
//! - ACB: `extract_acb_unique_to_memory(File)`, then per HCA waveform
//!   `HcaDecoder::from_reader(Cursor)` + `decode_to_pcm16_chunks` (FFmpeg FFI
//!   MP3 path) or `decode_to_wav` into a pre-sized `Vec` (WAV path).
//! - USM: `extract_usm_to_memory(File, name, None, export_audio)` on an
//!   unbuffered `File`, plus `read_metadata_file`.
//!
//! Usage:
//!   perf_corpus acb  <iters> <file.acb>...   decode throughput / real-time factor
//!   perf_corpus usm  <iters> <file.usm>...   demux throughput (File and in-memory)
//!   perf_corpus hca  <iters> <file.hca>...   raw HCA decode throughput
//!   perf_corpus digest <file>...             FNV-1a digests of every output
//!   perf_corpus once <acb|acbwav|usm|hca> <file>   one production-shaped run (peak RSS)
//!   perf_corpus rep <n> <acb|acbwav|usm|hca> <file> the same, n times (cycle counts)

use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::Cursor;
use std::path::Path;
use std::time::Instant;

use cridecoder::{extract_acb_unique_to_memory, extract_usm_to_memory, HcaDecoder};

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn decode_pcm16(hca: &[u8]) -> (u64, u64) {
    let mut d = HcaDecoder::from_reader(Cursor::new(hca)).expect("hca header");
    let mut n = 0u64;
    let mut acc = 0u64;
    d.decode_to_pcm16_chunks(|pcm| {
        n += pcm.len() as u64;
        acc = acc.wrapping_add(pcm[pcm.len() / 2] as u64);
        Ok(())
    })
    .expect("decode");
    (n, acc)
}

fn decode_wav(hca: &[u8]) -> Vec<u8> {
    let mut d = HcaDecoder::from_reader(Cursor::new(hca)).expect("hca header");
    let info = d.info();
    let total = (info.block_count * info.samples_per_block as u32)
        .saturating_sub(info.encoder_delay) as usize;
    let mut wav = Vec::with_capacity(44 + total * info.channel_count as usize * 2);
    d.decode_to_wav(&mut wav).expect("decode");
    wav
}

fn audio_seconds(hca: &[u8]) -> f64 {
    let d = HcaDecoder::from_reader(Cursor::new(hca)).expect("hca header");
    let i = d.info();
    (i.block_count as f64 * i.samples_per_block as f64) / i.sampling_rate as f64
}

fn acb_waveforms(path: &str) -> Vec<Vec<u8>> {
    let f = File::open(path).expect("open acb");
    extract_acb_unique_to_memory(f, Some(Path::new(path)))
        .expect("extract acb")
        .into_iter()
        .filter(|w| w.extension.eq_ignore_ascii_case("hca"))
        .map(|w| w.data)
        .collect()
}

/// Median wall time of `iters` runs (after one warm-up run), in seconds.
fn time<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    f();
    let mut samples: Vec<f64> = (0..iters.max(1))
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn bench_hca_set(label: &str, iters: usize, input_bytes: usize, hcas: &[Vec<u8>]) {
    let secs: f64 = hcas.iter().map(|h| audio_seconds(h)).sum();
    let hca_bytes: usize = hcas.iter().map(Vec::len).sum();
    let pcm = time(iters, || {
        for h in hcas {
            black_box(decode_pcm16(h));
        }
    });
    let wav = time(iters, || {
        for h in hcas {
            black_box(decode_wav(h));
        }
    });
    println!(
        "{label}: in={:.2}MB hca={:.2}MB audio={:.1}s tracks={} | pcm16 {:.2} ms ({:.0}x RT, {:.1} MB/s hca) | wav-vec {:.2} ms ({:.0}x RT)",
        input_bytes as f64 / 1e6,
        hca_bytes as f64 / 1e6,
        secs,
        hcas.len(),
        pcm * 1e3,
        secs / pcm,
        hca_bytes as f64 / pcm / 1e6,
        wav * 1e3,
        secs / wav,
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("");
    match mode {
        "acb" => {
            let iters: usize = args[2].parse().unwrap();
            for path in &args[3..] {
                let size = std::fs::metadata(path).unwrap().len() as usize;
                let extract = time(iters, || {
                    black_box(acb_waveforms(path));
                });
                let hcas = acb_waveforms(path);
                let name = Path::new(path).file_name().unwrap().to_string_lossy();
                println!(
                    "{name}: extract_acb_unique_to_memory {:.3} ms",
                    extract * 1e3
                );
                bench_hca_set(&name, iters, size, &hcas);
            }
        }
        "hca" => {
            let iters: usize = args[2].parse().unwrap();
            for path in &args[3..] {
                let data = std::fs::read(path).unwrap();
                let name = Path::new(path).file_name().unwrap().to_string_lossy();
                bench_hca_set(&name, iters, data.len(), std::slice::from_ref(&data));
            }
        }
        "usm" => {
            let iters: usize = args[2].parse().unwrap();
            for path in &args[3..] {
                let name = Path::new(path).file_name().unwrap().to_string_lossy();
                let size = std::fs::metadata(path).unwrap().len() as f64;
                let file_video = time(iters, || {
                    let f = File::open(path).unwrap();
                    black_box(extract_usm_to_memory(f, b"x.usm", None, false).unwrap());
                });
                let file_av = time(iters, || {
                    let f = File::open(path).unwrap();
                    black_box(extract_usm_to_memory(f, b"x.usm", None, true).unwrap());
                });
                let bytes = std::fs::read(path).unwrap();
                let mem_av = time(iters, || {
                    black_box(
                        extract_usm_to_memory(Cursor::new(&bytes[..]), b"x.usm", None, true)
                            .unwrap(),
                    );
                });
                let meta = time(iters, || {
                    black_box(cridecoder::usm::read_metadata_file(Path::new(path)).unwrap());
                });
                println!(
                    "{name}: {:.1}MB | File video-only {:.2} ms ({:.0} MB/s) | File a+v {:.2} ms ({:.0} MB/s) | Cursor a+v {:.2} ms ({:.0} MB/s) | metadata {:.3} ms",
                    size / 1e6,
                    file_video * 1e3,
                    size / file_video / 1e6,
                    file_av * 1e3,
                    size / file_av / 1e6,
                    mem_av * 1e3,
                    size / mem_av / 1e6,
                    meta * 1e3,
                );
            }
        }
        "digest" => {
            for path in &args[2..] {
                let lower = path.to_ascii_lowercase();
                if lower.ends_with(".acb") {
                    let f = File::open(path).unwrap();
                    for w in extract_acb_unique_to_memory(f, Some(Path::new(path))).unwrap() {
                        let cues: Vec<_> = w.cues.iter().map(|c| c.name.as_str()).collect();
                        println!(
                            "{path} wf {} {} {:016x} {:?}",
                            w.extension,
                            w.data.len(),
                            fnv1a(&w.data),
                            cues
                        );
                        if w.extension.eq_ignore_ascii_case("hca") {
                            let wav = decode_wav(&w.data);
                            println!("{path}   wav {} {:016x}", wav.len(), fnv1a(&wav));
                            let mut pcm = Vec::new();
                            let mut d = HcaDecoder::from_reader(Cursor::new(&w.data)).unwrap();
                            d.decode_to_pcm16_chunks(|c| {
                                pcm.extend_from_slice(c);
                                Ok(())
                            })
                            .unwrap();
                            let pcm_bytes: Vec<u8> =
                                pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
                            println!("{path}   pcm16 {} {:016x}", pcm.len(), fnv1a(&pcm_bytes));
                        }
                    }
                } else if lower.ends_with(".usm") {
                    for audio in [false, true] {
                        let f = File::open(path).unwrap();
                        for s in extract_usm_to_memory(f, b"x.usm", None, audio).unwrap() {
                            println!(
                                "{path} audio={audio} {}.{} {} {:016x}",
                                s.name,
                                s.extension,
                                s.data.len(),
                                fnv1a(&s.data)
                            );
                        }
                    }
                    let m = cridecoder::usm::read_metadata_file(Path::new(path)).unwrap();
                    // Round-trip through Value: its map is sorted, so HashMap order
                    // does not leak into the digest.
                    let json = serde_json::to_value(&m).unwrap().to_string();
                    println!("{path} metadata {:016x}", fnv1a(json.as_bytes()));
                } else if lower.ends_with(".hca") {
                    let data = std::fs::read(path).unwrap();
                    let wav = decode_wav(&data);
                    println!("{path} wav {} {:016x}", wav.len(), fnv1a(&wav));
                }
            }
        }
        "once" | "rep" => {
            let (reps, rest) = if mode == "rep" {
                (args[2].parse::<usize>().unwrap(), 3)
            } else {
                (1, 2)
            };
            let kind = args[rest].as_str();
            let path = &args[rest + 1];
            for _ in 0..reps {
                match kind {
                    "acb" => {
                        for h in acb_waveforms(path) {
                            black_box(decode_pcm16(&h));
                        }
                    }
                    "acbwav" => {
                        for h in acb_waveforms(path) {
                            black_box(decode_wav(&h));
                        }
                    }
                    "usm" => {
                        let f = File::open(path).unwrap();
                        black_box(extract_usm_to_memory(f, b"x.usm", None, false).unwrap());
                    }
                    "hca" => {
                        let data = std::fs::read(path).unwrap();
                        black_box(decode_pcm16(&data));
                    }
                    _ => panic!("unknown kind"),
                }
            }
        }
        _ => {
            eprintln!("usage: perf_corpus <acb|usm|hca|digest|once> ...");
            std::process::exit(2);
        }
    }
}
