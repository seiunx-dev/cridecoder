# AGENTS.md

## Project Overview

**cridecoder** is a pure Rust library for decoding CRI Middleware formats, with optional Python bindings via pyo3/maturin.

### Credits

CRI format implementation is based on [vgmstream](https://github.com/vgmstream/vgmstream) and [PyCriCodecs](https://github.com/Youjose/PyCriCodecs/).

## Architecture

```
src/
├── lib.rs          # Crate root, re-exports public API, #[pymodule] (behind "python" feature)
├── reader.rs       # Binary reader utilities (endianness, primitive types, alignment)
├── acb.rs          # ACB module root (re-exports submodules)
├── acb/
│   ├── consts.rs   # ACB/UTF constants and helpers
│   ├── utf.rs      # CRI UTF table parser
│   ├── afs.rs      # AFS2 archive parser
│   ├── track.rs    # Track list extraction from ACB
│   ├── extractor.rs # ACB extraction logic (disk + in-memory + de-duplicated)
│   ├── decode.rs   # High-level ACB → WAV decoding (extract + HCA decode + subkey)
│   └── builder.rs  # ACB/AWB/UTF table builders
├── hca.rs          # HCA module root (re-exports submodules)
├── hca/
│   ├── decoder.rs  # Core HCA decoder (ClHca, header parsing, block decoding)
│   ├── encoder.rs  # HCA encoder (PCM/WAV → HCA, optional encryption)
│   ├── hca_file.rs # High-level HcaDecoder with streaming, WAV output, key testing
│   ├── tables.rs   # Lookup tables for HCA decoding
│   ├── cipher.rs   # HCA encryption/decryption cipher
│   ├── ath.rs      # ATH (Absolute Threshold of Hearing) tables
│   ├── bitreader.rs # Bit-level reader/writer
│   └── imdct.rs    # Inverse MDCT transform
├── usm.rs          # USM module root (re-exports submodules)
├── usm/
│   ├── extractor.rs # USM extraction (video/audio stream demuxing)
│   ├── builder.rs   # USM container builder
│   └── metadata.rs  # USM metadata reading and JSON export
└── python.rs       # Python bindings (behind "python" feature)
```

## Key Types

ACB extraction has three flavors, each with a disk and an in-memory variant:

- `extract_acb_from_file()` / `extract_acb()` — extract tracks to a directory (returns written paths)
- `extract_acb_tracks_from_file()` / `extract_acb_tracks()` — same, plus per-track metadata (`ExtractedTrackFile` with `name`, `cue_id`, `subkey`)
- `extract_acb_to_memory()` — extract waveform bytes per cue without touching disk (`ExtractedAcbTrack`)
- `extract_acb_unique_to_memory()` — extract each physical waveform **once**, mapping shared cues onto it (`UniqueWaveform` + `AcbCueRef`); ACBs often point several cues at one waveform

High-level decode (extract + HCA decode in one call, per-AWB AFS2 subkey applied automatically):

- `decode_acb_to_wav_from_file()` / `decode_acb_to_wav()` — write decoded WAVs to a directory
- `decode_acb_to_wav_to_memory()` — return decoded `DecodedAcbTrack`s in memory; encrypted (type-56) ACBs need only the global keycode
- Each has a `*_parallel` variant taking a `threads` budget (tracks decoded concurrently, block-parallel HCA decoding for the leftover budget; output identical, `threads <= 1` is serial)

Other entry points:

- `AcbBuilder` / `TrackInput` — build ACB/AWB containers
- `HcaDecoder` — high-level HCA to WAV/PCM decoder; `HcaEncoder` — PCM/WAV to HCA
- `ClHca` — low-level HCA decoder state machine
- `extract_usm_file()` / `extract_usm()` — USM extraction (disk); `extract_usm_to_memory()` — in-memory; `UsmBuilder` — build USM

**Python bindings** (`src/python.rs`, behind the `python` feature) mirror this surface. Most disk functions have a `*_bytes` in-memory counterpart that takes/returns `bytes` via `Cursor` (no `.to_vec()` copy); the exceptions are `extract_acb` / `extract_acb_tracks`, which share `extract_acb_bytes` (embedded AWB only — external `.awb` needs the disk path), and `read_usm_metadata`, which is disk-only. The `.pyi` stubs in `cridecoder.pyi` are the source of truth for the Python signatures and must stay in sync with the bindings.

## Building

```bash
# Pure Rust
cargo build
cargo test                              # Unit tests only
RUST_MIN_STACK=16777216 cargo test       # Unit + integration tests (HCA needs larger stack)

# Python extension
python3 -m maturin build --release

# crates.io dry run
cargo publish --dry-run --allow-dirty
```

## Testing

- **Unit tests**: Inline `#[cfg(test)] mod tests` in most modules
- **Integration tests**: `tests/integration_tests.rs` — requires `se_0126_01.acb` and `0703.usm` test fixtures in project root
- Integration tests need `RUST_MIN_STACK=16777216` due to large `ClHca` struct

## Conventions

- Use `thiserror` for error types
- Use `byteorder` for binary reading via the `reader.rs` wrapper
- Use `encoding_rs` for Shift-JIS text decoding (CRI uses Shift-JIS strings)
- Public API lives in module root files (`acb.rs`, `hca.rs`, `usm.rs`); internals are `mod` (private)
- Python bindings are behind `#[cfg(feature = "python")]` so they're opt-in

## Git commits

All commit subjects must follow:

```text
[Type] Short description starting with capital letter
```

Allowed types:

| Type      | Usage                                                 |
|-----------|-------------------------------------------------------|
| `[Feat]`  | New feature or capability                             |
| `[Fix]`   | Bug fix                                               |
| `[Chore]` | Maintenance, refactoring, dependency or build changes |
| `[Docs]`  | Documentation-only changes                            |
| `[Perf]`  | Performance improvement (no behavior change)          |
| `[CI]`    | CI / release workflow changes                         |

Rules:

- Description starts with a capital letter.
- Use imperative mood: `Add ...`, not `Added ...`.
- No trailing period.
- Keep the subject at or below roughly 70 characters.
- **Agent attribution uses the standard Git `Co-authored-by:` trailer in the commit body, not a free-form `Agent:` line.** This makes GitHub render the co-author avatar on the commit page. The trailer must be on its own line, separated from the subject by a blank line, in the form `Co-authored-by: <Display Name> <email>`. Suggested values per agent:
  - Claude: `Co-authored-by: Claude <Model> <noreply@anthropic.com>` (substitute the actual model name, e.g. `Claude Opus 4.7`, `Claude Sonnet 4.6`)
  - Codex: `Co-authored-by: Codex <noreply@openai.com>`
  - Copilot: `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`

Examples from this repo's history:

```text
[Feat] Add encoding Python bindings
[Fix] Resolve check and clippy warnings
[Chore] Configure Dependabot updates
[Feat] Add encoding support but not tested in game
```

## GitHub Actions workflows

CI reuses the shared templates in
[`seiunx-dev/ci-templates`](https://github.com/seiunx-dev/ci-templates) at `@v1`.
The files in `.github/workflows` are thin callers:

- `ci.yml` (`CI`) runs on `main` pushes, pull requests targeting `main`, and manual
  dispatch:
  - `Rust` (`rust-ci`): `cargo fmt --check`, clippy `--all-targets -D warnings` for the
    default features and for `--features python`, and the tests run once under
    `cargo llvm-cov` (`RUST_MIN_STACK=16777216`).
  - `Wheel smoke` (`maturin-wheels`): builds one linux-x64 wheel and imports
    `cridecoder` from it, so the PyO3 bindings are compiled on every PR.
  - `Sonar` scans the coverage (skipped green on Dependabot/fork PRs); `Workflow lint`
    runs actionlint.
- The aggregate job **`CI OK`** is the only required status check.
- `release.yml` (`Release`) replaces the old `release-crate.yml` / `release-python.yml`
  (which rewrote the version from the tag with `sed`). Bump `version` in **both**
  `Cargo.toml` and `pyproject.toml` (and the package's own entry in `Cargo.lock`) in a PR
  → merge and wait for `CI OK` on `main` → push the tag `v<version>`. Pushing the tag
  creates the GitHub Release; do not create the Release by hand first.
  `release-gate` refuses a tag that differs from `Cargo.toml` and waits for `CI OK` on
  the tagged commit; then the wheels are built (not abi3: one wheel per interpreter,
  linux x64/arm64 in the manylinux container, macOS arm64/x64 and Windows x64 for
  3.9–3.14t) plus the sdist, the GitHub Release is published, and the wheels go to PyPI
  (trusted publishing, environment `pypi`) and the crate to crates.io (environment
  `crates-io`, `CARGO_REGISTRY_TOKEN`). Manual dispatch is a dry run: it builds the
  wheels and publishes nothing.
- CI never rewrites `Cargo.toml` / `pyproject.toml` or regenerates `Cargo.lock`.

Workflow maintenance rules:

- Use the shared templates first. Add custom jobs or steps only when a template
  genuinely cannot meet the project's needs, keep them in the thin caller files, and
  add a comment explaining why. The PyPI and crates.io publish jobs live in
  `release.yml` because trusted publishing is bound to the calling workflow file.
- Template bugs and missing features are fixed upstream in `seiunx-dev/ci-templates`
  (new `v1.x.y` tag), not worked around here.
- Keep top-level `permissions: contents: read`; grant `contents: write` / `id-token: write`
  only on the job that needs it.
- Do not set `sonar.projectVersion` or `*.reportPaths` in `sonar-project.properties`, and
  do not suppress `githubactions:S7637` there: the template's `sonar.yml` passes the
  version (`project-version: auto`) and report paths, and ignores S7637 for the `@v1`
  references.
- Third-party actions in caller-side custom steps are pinned to a full commit SHA with a
  `# vX.Y.Z` comment; Dependabot (`github-actions`) updates them and the template refs.
