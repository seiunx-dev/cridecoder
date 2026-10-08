# Known gaps / deferred items

These are confirmed divergences from CRI/vgmstream behavior that are **intentionally
not implemented yet**. All of them are **non-essential for the common decode/extract
path** (and specifically for Project Sekai assets, which use unencrypted HCA-MX in
ACB/AWB and VP9 USM). They are deferred because they are either pure performance,
encoder-only paths with nothing to validate against, or deep restructures with **no
test fixtures**, where a blind change to working code carries more regression risk
than value.

Verified against vgmstream `3f860bef` (`src/coding/libs/clhca.c`, `src/meta/acb.c`,
`src/meta/awb.c`) and PyCriCodecs `usm.py` / `acb.py`.

## ACB

### Command / Synth traversal breadth — `src/acb/track.rs`
`extract_tracks_from_event` / `extract_track_from_command` handle only the common path:
command `0x07d0` (noteOn 2000) → `u1 == 2` (Synth) → **first** ReferenceItem →
item-type 1 (Waveform). vgmstream (`acb.c:481-600`) also accepts noteOn `2003`, follows
`tlv_type == 3` (Sequence), iterates **all** ReferenceItems, and recurses on item-types
2 (Synth) and 3 (Sequence).
- **Impact:** ACBs whose cues fan out to multiple/ nested waveforms extract only the
  first. pjsk cues are single `noteOn → Synth → one Waveform`, so unaffected.
- **Why deferred:** rewriting the core recursive traversal can silently change output
  (extra/duplicate/mis-named tracks) on real multi-Synth/Sequence ACBs, and the test
  ACBs are builder-made single-cue — no fixture would catch a semantic regression.
  Needs a real multi-reference ACB fixture first.

### Type-8 BlockSequence — `src/acb/track.rs`
Cue `ReferenceType == 8` is currently routed through the Sequence path. vgmstream uses a
dedicated `load_acb_blocksequence` → `load_acb_block` over the Block/BlockSequence tables
(`acb.c:826-968`), which cridecoder does not parse. (Unknown ref types are now skipped
rather than erroring; landed in PR #10, `025e47e`.)
- **Impact:** none for pjsk (cues are type 3).
- **Why deferred:** requires new Block/BlockSequence table parsing; implementable from
  vgmstream but unverifiable without a type-8 fixture.

### Legacy `Id` waveform field — `src/acb/track.rs`
`extract_track_from_command` selects the waveform id from `StreamAwbId`/`MemoryAwbId`
keyed on the `Streaming` flag. vgmstream (`acb.c:335-363`) first tries the legacy single
`Id` column, falling back by `acb->is_memory`.
- **Impact:** old ACBs that carry only `Id` (no split ids) resolve to id 0. pjsk uses
  split ids, so unaffected.
- **Why deferred:** `get_int_field` returns 0 for an absent column (can't distinguish
  absent from 0); doing this correctly needs an optional column reader. Additive and
  low-risk, but the legacy path has no fixture.

## USM

### Per-channel (`chno`) multi-track — `src/usm/extractor.rs`
The `chno` byte (chunk offset 0x0C) is read but not used to separate streams; all `@SFA`
chunks merge into one audio output. vgmstream/PyCriCodecs key outputs by
`<signature>_<chno>`.
- **Impact:** multi-track audio interleaves into one corrupt file. pjsk USMs are single
  video + single audio (`chno == 0`), so unaffected.
- **Why deferred:** requires restructuring the output path to per-`chno` sinks; the
  single-track path is testable but multi-track has no fixture.

## HCA encoder (PCM → HCA; not used for decoding/extraction)

### Loop frame alignment (P17) — `src/hca/encoder.rs`
The encoder does not apply CRI's loop pipeline (delay bump, 2048-byte loop-frame
alignment, post-loop tail) from `clhca.c`, so a loop encoded by cridecoder would loop at
the wrong sample in clHCA players.
- **Status:** `HcaEncoderConfig` already has public `loop_start` / `loop_end` fields.
  When both are set, the encoder writes a `loop` header chunk that only offsets the
  points by the encoder delay, without CRI's alignment, so those loop points are **not
  CRI-accurate**. The Python `encode_hca*` bindings do not expose them.
- **Why deferred:** no test sets encoder loop points and there is no clHCA-player
  check to validate a CRI-accurate pipeline against; encoding is not on the pjsk path.

### MDCT is O(N²) — `src/hca/encoder.rs`
`mdct_transform` computes the forward DCT-IV as a naive double loop (`cos()` per (n,k)
pair, ~16384 calls/subframe). It is mathematically correct (round-trips), just slow. The
**decoder** side (`imdct.rs`) already uses a factorized DCT-IV with precomputed
`SIN_TABLES`/`COS_TABLES`; the encoder could reuse the same approach (O(N log N)).
- **Impact:** encoding speed only. Decoding (the pjsk path) is unaffected.
- **Why deferred:** bit-sensitive rewrite; the existing round-trip tests only assert
  non-empty output, so a proper encode→decode→compare-to-input fidelity test should be
  added as a guard before changing it.

## Intentionally not changed

### ms_stereo permissiveness — `src/hca/decoder.rs`
cridecoder decodes ms_stereo HCA; vgmstream (`clhca.c:985`) rejects it as
`HCA_ERROR_HEADER` (`//TODO: should work but untested`). cridecoder's behavior is the
more permissive (and arguably more useful) one, so this divergence is **kept on purpose**.
