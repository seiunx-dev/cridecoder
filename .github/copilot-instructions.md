# Copilot Instructions for cridecoder

[`AGENTS.md`](../AGENTS.md) is the single source of truth for this repository; if this file
and AGENTS.md disagree, follow AGENTS.md.

## Project Context

This is a pure Rust library (`cridecoder`) for decoding CRI Middleware audio/video formats (ACB, HCA, USM). The CRI format implementation is based on [vgmstream](https://github.com/vgmstream/vgmstream) and [PyCriCodecs](https://github.com/Youjose/PyCriCodecs/). It also provides optional Python bindings via pyo3/maturin.

## Code Style

- **Rust edition**: 2021
- **Error handling**: Use `thiserror` derive macros for error enums
- **Binary I/O**: Use the `reader.rs` wrapper around `byteorder` for all binary reading
- **Text encoding**: CRI formats use Shift-JIS; use `encoding_rs::SHIFT_JIS` for decoding
- **Module structure**: Use Rust 2018+ flat module style (`src/acb.rs` + `src/acb/` directory), not `mod.rs`
- **Feature gates**: Python bindings use `#[cfg(feature = "python")]` — pure Rust builds should not depend on pyo3
- **Tests**: Unit tests are inline `#[cfg(test)] mod tests`, integration tests are in `tests/`

## Important Notes

- The `ClHca` struct (HCA decoder state) is very large (~200KB on stack). Use `RUST_MIN_STACK=16777216` when running integration tests
- ACB files may contain embedded AWB data or reference external `.awb` files
- HCA files support encryption — use `HcaDecoder::set_encryption_key(keycode, subkey)`, or `HcaDecoder::test_key()` with a `KeyTest` for key testing (`ClHca::set_key()` is the low-level equivalent)
- USM files interleave video (`@SFV`: VP9 → `.ivf`, otherwise MPEG2 → `.m2v`) and audio (`@SFA`: ADX or HCA) chunks; the USM audio XOR mask applies only to ADX, since HCA has its own cipher

## Public API

```rust
// ACB
pub fn extract_acb_from_file(path, output_dir) -> Result<Option<Vec<String>>>
pub fn extract_acb(reader, output_dir, acb_file_path: Option<&Path>) -> Result<Vec<String>>  // path locates an external .awb

// HCA
pub struct HcaDecoder { ... }
impl HcaDecoder {
    pub fn from_file(path) -> Result<Self>
    pub fn from_reader(reader) -> Result<Self>
    pub fn info(&self) -> &HcaInfo
    pub fn decode_to_wav(&mut self, writer) -> Result<()>
    pub fn decode_all(&mut self) -> Result<Vec<f32>>
}

// USM
pub fn extract_usm_file(path, output_dir, key, export_audio) -> Result<Vec<PathBuf>>
pub fn extract_usm(reader, output_dir, name, key, export_audio) -> Result<Vec<PathBuf>>
```

## Testing

```bash
cargo test                              # Unit tests only
RUST_MIN_STACK=16777216 cargo test       # Full test suite (needs test fixture files)
```

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
