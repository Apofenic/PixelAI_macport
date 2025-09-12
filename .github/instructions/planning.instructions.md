# PixelAI Planning Instructions

> NOTE: Intended to follow project convention of using a template from user templates directory. Template files were not accessible from current workspace context, so this plan was generated from the audited architecture. Replace/merge with official template boilerplate if required.

## 1. Objective

Add first‑class macOS (Apple Silicon) support with: (a) proper device detection (CUDA / MPS / CPU), (b) an MLX inference backend for Stable Diffusion style pixel art generation (replacing current placeholder), (c) clean backend abstraction, and (d) UI + API adjustments for engine capabilities, while maintaining existing Torch (diffusers) functionality and LoRA workflows.

## 2. Key Outcomes (Definition of Done)

1. Server auto-detects cuda -> mps -> cpu and logs chip / capability.
2. MLX backend: real latent diffusion (or clearly documented experimental stage) producing non‑placeholder images.
3. Unified backend interface (TorchBackend / MLXBackend) with consistent generate() contract.
4. `/generate` optionally accepts `engine`; response includes `engine` used.
5. New endpoints: `/engines` (capability matrix) and optional `/mlx_models` if naming diverges.
6. Lua UI dynamically enables/disables LoRA + model controls per selected engine; LoRA disabled (or error) if unsupported in MLX MVP.
7. Error taxonomy expanded (MODEL_NOT_CONVERTED, LORA_UNSUPPORTED_ENGINE, MPS_OOM, etc.).
8. Conversion script for diffusers → MLX weights in `scripts/` with README instructions.
9. Minimal smoke tests for torch + mlx (if installed) generation at 64×64.
10. Documentation updated (README + extension description) including installation, limitations, fallback behavior.

## 3. Constraints & Assumptions

- Assume Python 3.10+ environment with ability to install `mlx`, `torch`, `diffusers`, `transformers`.
- Initial MLX release may skip LoRA; schedule LoRA-on-MLX as later phase.
- Background removal (BiRefNet) remains Torch-only; optional future MLX adaptation.
- Large model conversions may be user-initiated (not automatic) to avoid long startup blocking.

## 4. Architecture Changes

Current: Monolithic `PixelArtSDServer` directly invoking diffusers pipeline + a stub MLX method.

Target: Introduce `backends/` package:

- `base_backend.py` – abstract class (ensure_loaded, generate, supports(feature), unload_lora(optional)).
- `torch_backend.py` – existing logic refactored out of server class.
- `mlx_backend.py` – real implementation (or staged), loading converted weights from `models/mlx/<model_id>`.

`sd_server.py` becomes orchestration layer: selects backend, uniform validation, post-processing, background removal, error classification.

## 5. Data / Model Handling

Torch models: unchanged (diffusers cached in ~/.cache/huggingface or local).
MLX models: store converted arrays (e.g., `.npz` or directory of `.npy`) under `models/mlx/<safe_name>/` with manifest JSON recording original HF revision and hash for cache integrity.

## 6. API Evolution

Add:

- `GET /engines` → { engines: { torch: {...}, mlx: {...} } }
- `POST /set_engine` (exists) – keep; encourage per-request engine override instead.

Modify:

- `POST /generate` accepts optional `engine`; server uses specified or current default.

Response includes `engine` and explicit `model_name`.

## 7. Error & Logging Strategy

Extend classify to map:

- MODEL_NOT_CONVERTED (MLX weights missing)
- LORA_UNSUPPORTED_ENGINE
- MPS_OOM (detect mps + memory phrases)
- ENGINE_INTERNAL (generic backend failure)

Structured log prefix: `[GEN][torch]` / `[GEN][mlx]` with timing and resolution.

## 8. Performance Considerations

Torch: enable attention/vae slicing by default; consider user flag to disable for max speed.
MLX: reuse latent buffers; prefer fp16 where stable; fallback fp32 if numeric instability.
Early dimension guard (reject > 2048).

## 9. Security / Safety

Whitelist model names via config set or pattern; reject path traversal (`..`, absolute paths) in model or LoRA parameters.
Gate auto pip installs behind `ALLOW_AUTO_INSTALL` env or CLI flag.

## 10. Phased Implementation Roadmap

Phase 1: Device detection + refactor Torch backend (stable baseline).
Phase 2: Backend abstraction + routing; add `/engines` endpoint.
Phase 3: MLX backend MVP (simple inference or minimal diffusion loop) + conversion script.
Phase 4: UI adjustments (engine-aware controls, LoRA disable, hints).
Phase 5: Error taxonomy expansion + logging improvements.
Phase 6: Documentation + tests + README update.
Phase 7 (optional): LoRA-on-MLX exploration; background removal modularization.

## 11. Testing Plan

Smoke tests: torch generation 64×64; mlx fallback (skip if mlx missing) producing deterministic seed.
Failure tests: request MLX without conversion → MODEL_NOT_CONVERTED; LoRA with MLX → LORA_UNSUPPORTED_ENGINE.
Manual acceptance: UI toggles, generation latency logs, error dialogs.

## 12. Risks & Mitigations

MLX instability / API changes → isolate in single module, version-check at load.
User confusion about experimental engine → explicit labeling + README warnings.
Memory spikes on MPS → default smaller resolution suggestion in warning.

## 13. Documentation Updates (To Produce)

- README: Installation (Torch + optional MLX), weight conversion usage, engine differences table, troubleshooting.
- Extension description: “Experimental MLX backend on Apple Silicon.”
  
Keep a CHANGELOG entry describing new endpoints / flags.

## 14. Success Metrics (Qualitative)

- Users can switch engines without restarting.
- MLX path produces non-placeholder images.
- Clear actionable errors for unsupported operations.
- Reasonable parity in small (512×512) generation latency vs Torch MPS.

## 15. Follow-Up (Deferred Items)

- Progressive preview / step streaming.
- LoRA merging or adaptation for MLX.
- Quantized weight option for memory savings.
- WebSocket progress updates.

---
Revision 1
