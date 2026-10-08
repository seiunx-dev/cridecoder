# cridecoder

A pure Rust library for CRI Middleware codec encoding and decoding. Supports ACB/AWB audio containers, HCA (High Compression Audio) encoding/decoding, and USM video container extraction/building.

## Credits

This project's CRI format implementation is based on and inspired by:

- [vgmstream](https://github.com/vgmstream/vgmstream) — a library for playing streamed audio from video games.
- [PyCriCodecs](https://github.com/Youjose/PyCriCodecs/) — a Python library for CRI Middleware codecs.

Many thanks to the vgmstream and PyCriCodecs contributors for their reverse-engineering work on CRI Middleware formats.

## Features

- **ACB/AWB Extraction & Building** — Parse and create CRI ACB audio containers
- **ACB → WAV in one call** — Extract and decode an ACB straight to WAV; the per-AWB AFS2 subkey is applied automatically for encrypted (type-56) HCA
- **In-memory APIs** — Every extract/decode/build operation has a variant that works on `bytes`/`Vec<u8>` without touching disk
- **De-duplicated extraction** — Emit each physical waveform once even when several cues share it
- **HCA Encoding & Decoding** — Encode PCM to HCA, decode HCA to PCM/WAV
- **USM Extraction & Building** — Extract or create USM video containers
- **USM Metadata** — Read and export USM metadata as structured JSON
- **Key Testing** — Test decryption keys for encrypted HCA files
- **Encryption Support** — Encode HCA with encryption keys
- **Python bindings** — `pip install cridecoder` exposes the full API via PyO3
- **Pure Rust** — No C dependencies, works on any platform Rust supports

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
cridecoder = "0.3"
```

### ACB Extraction

```rust
use std::path::Path;
use cridecoder::extract_acb_from_file;

let tracks = extract_acb_from_file(
    Path::new("audio.acb"),
    Path::new("output/"),
).unwrap();

if let Some(tracks) = tracks {
    for track in &tracks {
        println!("Extracted: {}", track);
    }
}
```

### ACB → WAV (decode in one call)

Extract an ACB and decode its HCA tracks straight to WAV. The per-AWB AFS2
subkey is applied automatically, so encrypted (type-56) ACBs only need the
global keycode (pass `None` for unencrypted ACBs):

```rust
use std::path::Path;
use cridecoder::decode_acb_to_wav_from_file;

let paths = decode_acb_to_wav_from_file(
    Path::new("audio.acb"),
    Path::new("output/"),
    None, // Option<u64> global keycode
).unwrap();

for path in &paths {
    println!("Wrote: {path}");
}
```

### In-memory extraction (no disk I/O)

`extract_acb_to_memory` returns the waveform bytes per cue. For ACBs where
several cues share one physical waveform, `extract_acb_unique_to_memory`
emits each waveform once and lists the cues that reference it:

```rust
use std::io::Cursor;
use cridecoder::extract_acb_unique_to_memory;

let acb = std::fs::read("audio.acb").unwrap();
let waveforms = extract_acb_unique_to_memory(Cursor::new(acb), None).unwrap();

for wf in &waveforms {
    let cue_names: Vec<_> = wf.cues.iter().map(|c| c.name.as_str()).collect();
    println!("{} bytes ({}) → {:?}", wf.data.len(), wf.extension, cue_names);
}
```

### ACB Building

```rust
use std::io::Cursor;
use cridecoder::{AcbBuilder, TrackInput};

let hca_data = std::fs::read("track.hca").unwrap();
let track = TrackInput::new("my_track", 0, hca_data);

let mut builder = AcbBuilder::new();
builder.add_track(track);

let mut output = Cursor::new(Vec::new());
builder.build(&mut output, None).unwrap();
```

### HCA to WAV

```rust
use std::fs::File;
use cridecoder::HcaDecoder;

let mut decoder = HcaDecoder::from_file("audio.hca").unwrap();
let info = decoder.info();
println!("Sample rate: {}, Channels: {}", info.sampling_rate, info.channel_count);

let mut output = File::create("output.wav").unwrap();
decoder.decode_to_wav(&mut output).unwrap();
```

### PCM to HCA

```rust
use std::io::Cursor;
use cridecoder::{HcaEncoder, HcaEncoderConfig};

// Generate or load PCM samples (interleaved stereo f32)
let samples: Vec<f32> = vec![0.0; 44100 * 2]; // 1 second of silence

let config = HcaEncoderConfig::new(44100, 2)  // 44.1kHz stereo
    .with_bitrate(256_000);  // 256 kbps

let mut encoder = HcaEncoder::new(config).unwrap();
let mut output = Cursor::new(Vec::new());
encoder.encode(&samples, &mut output).unwrap();
```

### USM Extraction

```rust
use std::path::Path;
use cridecoder::extract_usm_file;

let files = extract_usm_file(
    Path::new("video.usm"),
    Path::new("output/"),
    None,   // optional video decryption key
    false,  // export audio
).unwrap();

for file in &files {
    println!("Extracted: {:?}", file);
}
```

### USM Building

```rust
use std::io::Cursor;
use cridecoder::UsmBuilder;

let video_data = std::fs::read("video.m2v").unwrap();

let builder = UsmBuilder::new("my_video".to_string())
    .video(video_data);

let mut output = Cursor::new(Vec::new());
builder.build(&mut output).unwrap();
```

## Python

The same API is available from Python via PyO3 bindings:

```bash
pip install cridecoder
```

```python
import cridecoder

# Extract + decode an ACB to WAV bytes in one call (no disk I/O).
# Encrypted (type-56) ACBs only need the global key; pass key=None otherwise.
acb = open("audio.acb", "rb").read()
for track in cridecoder.decode_acb_to_wav_bytes(acb, key=None):
    print(track["name"], track["extension"], len(track["data"]))

# De-duplicated extraction: each physical waveform once, with its cues.
for wf in cridecoder.extract_acb_unique_bytes(acb):
    print(wf["extension"], [c["name"] for c in wf["cues"]])

# Decode a standalone HCA (key/subkey for encrypted files).
wav = cridecoder.decode_hca_bytes(open("audio.hca", "rb").read())
```

Most disk-based functions have an in-memory `*_bytes` counterpart that takes and
returns `bytes` (`extract_acb` / `extract_acb_tracks` share `extract_acb_bytes`;
`read_usm_metadata` is disk-only). See `cridecoder.pyi` for the full typed signatures.

## Supported Formats

| Format | Description | Operations |
|--------|-------------|------------|
| ACB | CRI Audio Container | Extract / Build |
| AWB | CRI Audio Waveform Bank | Extract / Build |
| HCA | High Compression Audio | Encode / Decode |
| USM | CRI Video Container | Extract / Build |

## License

MIT
