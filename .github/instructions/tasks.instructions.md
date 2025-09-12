# PixelAI Task Instructions

> NOTE: Generated due to missing access to shared template directory. Align formatting with official template in a future pass if needed.

## Usage

Mark tasks as you proceed. Add emergent work under "Discovered During Work". Keep statuses: TODO / IN_PROGRESS / BLOCKED / DONE.

## Core Task List

### 1. Device & Platform Enhancements

- [ ] Add cuda/mps/cpu detection logic
- [ ] Log Apple chip + memory info
- [ ] Adjust precision rules (avoid unsafe fp16 on MPS)

### 2. Backend Abstraction

- [ ] Create `backends/` package and base interface
- [ ] Extract current Torch logic into `torch_backend.py`
- [ ] Wire server to delegate through registry

### 3. MLX Backend MVP

- [ ] Implement `mlx_backend.py` scaffold
- [ ] Weight loading (converted cache)
- [ ] Minimal diffusion loop (or placeholder replaced with real latent pipeline)
- [ ] Deterministic seeding
- [ ] Capability flag `supports_lora = False`

### 4. Weight Conversion Tooling

- [ ] Script `scripts/convert_to_mlx.py`
- [ ] Manifest output & integrity hash
- [ ] README section documenting usage

### 5. API Additions / Modifications

- [ ] `/engines` endpoint (capabilities)
- [ ] Extend `/generate` to accept `engine`
- [ ] Include `engine` in `/generate` response
- [ ] Guard dimension limits (<= 2048)

### 6. Error Handling Expansion

- [ ] New codes: MODEL_NOT_CONVERTED, LORA_UNSUPPORTED_ENGINE, MPS_OOM, ENGINE_INTERNAL
- [ ] Update classify_exception mapping
- [ ] Lua UI friendly message mapping

### 7. LoRA Logic Separation

- [ ] Conditional disable in UI when engine lacks LoRA
- [ ] Server rejection with clear error code

### 8. Lua UI Updates

- [ ] Enable engine-specific controls without disabling model selection for MLX
- [ ] Show capability hint (LoRA unsupported on MLX MVP)
- [ ] Dynamic disable LoRA dropdown & strength slider for MLX
- [ ] Display engine in status (already present; verify after changes)

### 9. Logging & Observability

- [ ] Prefix log lines with backend tag
- [ ] Log generation time, resolution, steps
- [ ] Optional JSON log mode flag

### 10. Performance / Memory

- [ ] Enable attention & VAE slicing by default (Torch)
- [ ] Add flag to disable slicing
- [ ] Reuse latent buffers MLX

### 11. Background Removal Modularization (Optional Phase)

- [ ] Move BiRefNet into separate module file
- [ ] Flag to disable loading (`--no-bg`)

### 12. Testing

- [ ] Add `tests/smoke_generate.py`
- [ ] Test Torch generation 64×64
- [ ] Test MLX path (skip gracefully if not installed)
- [ ] Negative tests for MLX missing weights & LoRA unsupported

### 13. Documentation

- [ ] README: MLX setup + conversion
- [ ] Capabilities table (Torch vs MLX)
- [ ] Troubleshooting section (OOM, missing conversion, install commands)
- [ ] Update extension description / version bump

### 14. Tooling & Scripts

- [ ] `scripts/verify_env.py` diagnostics
- [ ] Update `start_server.sh` to print engine guidance

### 15. Release Prep

- [ ] Version bump (server + extension)
- [ ] CHANGELOG entry
- [ ] Sanity run on macOS (M1/M2) & one CUDA machine

## Discovered During Work

Add new tasks here.

## Blockers

Log any blocking issues + mitigation here.

## Completed

Move DONE tasks here with brief note/date.

## Conventions

- Keep task wording imperative ("Add", "Implement", "Update").
- Avoid large hidden scope—split if > ~4 hours.
- Prefer PRs grouping related subtasks per phase.

---
Revision 1
