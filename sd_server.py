#!/usr/bin/env python3
"""
Local AI Generator Server for Aseprite
Professional pixel art generation using Stable Diffusion with LoRA support and BiRefNet background removal.
Version 2.0 - Enhanced and optimized for publishing
"""
import os
import sys
import json
import base64
import io
import warnings
from datetime import datetime, timezone, timedelta
import threading
from uuid import UUID, uuid4
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")
warnings.filterwarnings("ignore", message=".*CLIPTextModelWithProjection.*")

# Suppress Flask development server warning
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

class PixelArtSDServer:
    def __init__(self):
        # Core pipeline/model state
        self.pipeline = None
        self.segmentation_model = None
        self.segmentation_processor = None
        self.model_loaded = False
        self.current_model = None
        self.current_engine = 'torch'  # 'torch' or 'mlx'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cache = {}
        self.offline_mode = False

        # Progress tracking (Task 1.1.1)
        self.generation_progress = {}
        self.current_generation_id = None
        self.progress_lock = threading.RLock()
        self.progress_retention_seconds = 10
        self.progress_error_retention_seconds = 30
        self.enforce_monotonic_progress = True
        # Cancellation tracking
        self._cancellation_flags = set()
        # Expiry / retention enhancements
        self.expiry_grace_seconds = 5

        # Default generation settings
        self.default_settings = {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality",
            "pixel_art_prompt_suffix": ", pixel art, 8bit style, game sprite"
        }

        # Startup diagnostics
        print("🚀 Local AI Generator Server v2.0")
        print(f"📱 Device: {self.device}")
        print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

class CancelledGeneration(Exception):
    pass

    def load_segmentation_model(self):
        """Loads the BiRefNet model for professional background removal."""
        if self.segmentation_model and self.segmentation_processor:
            return True
            
        print("📦 Loading BiRefNet model for background removal...")
        
        # Check and install required packages
        missing_packages = []
        try:
            import einops
        except ImportError:
            missing_packages.append("einops")
            
        try:
            import kornia
        except ImportError:
            missing_packages.append("kornia")
        
        # Auto-install missing packages
        if missing_packages:
            print(f"📥 Installing missing packages: {', '.join(missing_packages)}")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print(f"✅ Successfully installed: {', '.join(missing_packages)}")
                
                # Re-import after installation
                import einops
                import kornia
            except Exception as e:
                print(f"❌ Failed to auto-install packages: {e}")
                print(f"💡 Please manually run: pip install {' '.join(missing_packages)}")
                return False
        
        try:
            model_name = 'zhengpeng7/BiRefNet'
            
            # Create processor for image preprocessing
            self.segmentation_processor = transforms.Compose([
                transforms.Resize((352, 352), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            # Load the segmentation model
            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.offline_mode
            )
            self.segmentation_model.to(self.device)
            self.segmentation_model.eval()
            
            print("✅ BiRefNet model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading BiRefNet model: {e}")
            return False

    def remove_background(self, pil_image):
        """Uses BiRefNet to create a high-quality transparency mask."""
        if not self.load_segmentation_model():
            raise Exception("Background removal model could not be loaded.")
            
        print("🎭 Removing background with BiRefNet...")
        try:
            with torch.no_grad():
                # Convert to RGB for processing
                rgb_image = pil_image.convert("RGB")
                
                # Preprocess image
                input_tensor = self.segmentation_processor(rgb_image).unsqueeze(0).to(self.device)
                
                # Generate mask
                outputs = self.segmentation_model(input_tensor)
                logits = outputs[0]
                
                # Resize mask to original image size
                mask = F.interpolate(logits, size=pil_image.size[::-1], mode='nearest')
                mask = torch.sigmoid(mask).squeeze()
                
                # Create binary mask with sharp edges for pixel art
                binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8)
                
            # Apply mask to create transparent background
            mask_image = Image.fromarray(binary_mask * 255, mode='L')
            rgba_image = pil_image.convert("RGBA")
            rgba_image.putalpha(mask_image)
            
            print("✅ Background removal complete")
            return rgba_image
            
        except Exception as e:
            print(f"❌ Error during background removal: {e}")
            return pil_image.convert("RGBA")

    def image_to_base64(self, image):
        """Convert PIL image to base64 encoded bytes."""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return base64.b64encode(image.tobytes()).decode()

    def load_model(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        """Load and cache AI models for efficient generation."""
        try:
            local_only = self.offline_mode
            if local_only:
                print("🔒 Offline mode enabled: loading from cache only")
                
            # Check cache first
            if model_name in self.model_cache:
                print(f"⚡ Loading {model_name} from cache...")
                self.pipeline = self.model_cache[model_name]
                self.current_model = model_name
                self.model_loaded = True
                return True
            
            print(f"📥 Loading base model: {model_name}")
            precision = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load appropriate pipeline based on model type
            if "xl" in model_name.lower():
                print("🔧 Loading SDXL pipeline with optimized VAE...")
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix",
                    torch_dtype=precision,
                    local_files_only=local_only
                )
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    vae=vae,
                    torch_dtype=precision,
                    use_safetensors=True,
                    local_files_only=local_only
                )
            else:
                print("🔧 Loading SD 1.5 pipeline...")
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=precision,
                    use_safetensors=True,
                    local_files_only=local_only
                )
            
            # Move to device and optimize
            self.pipeline = self.pipeline.to(self.device)
            
            # Cache the model for future use
            self.model_cache[model_name] = self.pipeline
            self.current_model = model_name
            self.model_loaded = True
            
            print(f"✅ Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {str(e)}")
            return False

    def generate_image(self, prompt, lora_model=None, lora_strength=1.0, **kwargs):
        """Generate images with LoRA style control and optimized settings."""
        if not self.model_loaded:
            raise Exception("No base model loaded. Please load a model first.")
        if self.current_engine == 'mlx':
            # Route to MLX placeholder implementation
            return self.generate_image_mlx(prompt, **kwargs)
        print(f"🎨 Generating image with prompt: '{prompt[:50]}...'")

        pipeline_kwargs = {}

        # Handle LoRA loading with strength control
        if lora_model and lora_model.lower() not in ['none', '']:
            print(f"🎭 Loading LoRA: {lora_model} (strength: {lora_strength})")
            try:
                if os.path.exists(lora_model):
                    lora_path, weight_name = os.path.split(lora_model)
                    self.pipeline.load_lora_weights(lora_path, weight_name=weight_name)
                else:
                    self.pipeline.load_lora_weights(lora_model)
                pipeline_kwargs["cross_attention_kwargs"] = {"scale": float(lora_strength)}
            except Exception as le:
                msg = str(le)
                if ("Target modules" in msg or "target modules" in msg or "size mismatch" in msg):
                    print("⚠️ LoRA appears incompatible with current base model. Full exception below:")
                    import traceback; traceback.print_exc()
                    raise Exception("INCOMPATIBLE_LORA")
                raise

        gen_params = self.default_settings.copy()
        gen_params.update(kwargs)

        if "width" not in kwargs or "height" not in kwargs:
            if "xl" in self.current_model.lower():
                gen_params.setdefault('width', 1024)
                gen_params.setdefault('height', 1024)
            else:
                gen_params.setdefault('width', 512)
                gen_params.setdefault('height', 512)

        if "pixel art" not in prompt.lower():
            prompt += gen_params["pixel_art_prompt_suffix"]

        seed = gen_params.get("seed", -1)
        generator = torch.Generator(device=self.device)
        if seed is not None and int(seed) != -1:
            generator.manual_seed(int(seed))
            print(f"🎲 Using seed: {seed}")
        else:
            import random
            random_seed = random.randint(0, 2**32 - 1)
            generator.manual_seed(random_seed)
            print(f"🎲 Using random seed: {random_seed}")
            seed = random_seed

        pipeline_kwargs.update({
            "prompt": prompt,
            "negative_prompt": gen_params["negative_prompt"],
            "width": gen_params["width"],
            "height": gen_params["height"],
            "num_inference_steps": int(gen_params["num_inference_steps"]),
            "guidance_scale": float(gen_params["guidance_scale"]),
            "generator": generator
        })

        print(f"⚙️ Generation settings: {gen_params['width']}x{gen_params['height']}, {gen_params['num_inference_steps']} steps")

        try:
            result = self.pipeline(**pipeline_kwargs)
            print("✅ Image generation complete")
            return result.images[0], generator.initial_seed()
        except Exception as gen_err:
            if str(gen_err) == "INCOMPATIBLE_LORA":
                raise Exception("INCOMPATIBLE_LORA")
            raise
        finally:
            if lora_model and lora_model.lower() not in ['none', ''] and hasattr(self.pipeline, 'unload_lora_weights'):
                try:
                    self.pipeline.unload_lora_weights()
                except Exception:
                    pass

    def generate_image_with_progress(self, generation_id: str, prompt, lora_model=None, lora_strength=1.0, progress_phases=None, **kwargs):
        """Progress-aware generation.

        Emits coarse + step-level progress updates via update_progress.
        progress_phases (optional) allows overriding default phase percentage milestones.
        """
        if not self.model_loaded:
            raise Exception("No base model loaded. Please load a model first.")
        if self.current_engine == 'mlx':
            # Use existing MLX stub (no internal steps yet)
            img, seed = self.generate_image_mlx(prompt, **kwargs)
            self.update_progress(generation_id, progress=100.0, status='completed', message='MLX placeholder complete')
            return img, seed

        # Default phase anchors (will be supplemented by actual diffusion steps later)
        # Phase anchors limited to those reached during diffusion window (<=75%).
        # Post-diffusion phases (80,90,95,98) are emitted explicitly after pipeline run.
        default_phases = progress_phases or [
            (0.0, 'Initializing'),
            (10.0, 'Preparing pipeline'),
            (25.0, 'Encoding prompt'),
            (40.0, 'Starting diffusion'),
            (60.0, 'Mid diffusion'),
            (75.0, 'Finishing diffusion steps'),
        ]

        total_steps = int(kwargs.get('num_inference_steps', self.default_settings['num_inference_steps']))
        width = kwargs.get('width')
        height = kwargs.get('height')

        # Baseline initialization update (settings already included by caller but ensure)
        self.update_progress(
            generation_id,
            message='Starting generation',
            progress=0.0,
            width=width,
            height=height,
            steps=total_steps,
            status='generating'
        )

        import time, math
        start_monotonic = time.monotonic()
        step_durations = []

        # Prepare pipeline args similar to generate_image
        gen_params = self.default_settings.copy()
        gen_params.update(kwargs)
        if "width" not in kwargs or "height" not in kwargs:
            if "xl" in self.current_model.lower():
                gen_params.setdefault('width', 1024)
                gen_params.setdefault('height', 1024)
            else:
                gen_params.setdefault('width', 512)
                gen_params.setdefault('height', 512)

        if "pixel art" not in prompt.lower():
            prompt_eff = prompt + gen_params["pixel_art_prompt_suffix"]
        else:
            prompt_eff = prompt

        seed = gen_params.get('seed', -1)
        generator = torch.Generator(device=self.device)
        if seed is not None and int(seed) != -1:
            generator.manual_seed(int(seed))
        else:
            import random
            seed = random.randint(0, 2**32 - 1)
            generator.manual_seed(seed)

        pipeline_kwargs = {
            "prompt": prompt_eff,
            "negative_prompt": gen_params["negative_prompt"],
            "width": gen_params['width'],
            "height": gen_params['height'],
            "num_inference_steps": int(gen_params["num_inference_steps"]),
            "guidance_scale": float(gen_params["guidance_scale"]),
            "generator": generator
        }

        # LoRA handling (duplicated minimal logic; could refactor later)
        if lora_model and lora_model.lower() not in ['none', '']:
            try:
                if os.path.exists(lora_model):
                    lora_path, weight_name = os.path.split(lora_model)
                    self.pipeline.load_lora_weights(lora_path, weight_name=weight_name)
                else:
                    self.pipeline.load_lora_weights(lora_model)
                pipeline_kwargs["cross_attention_kwargs"] = {"scale": float(lora_strength)}
            except Exception as le:
                msg = str(le)
                if ("Target modules" in msg or "target modules" in msg or "size mismatch" in msg):
                    raise Exception("INCOMPATIBLE_LORA")
                raise

        # Inform about starting diffusion
        self.update_progress(generation_id, message='Initializing diffusion', progress=5.0)

        # Hook into pipeline progress (diffusers allows callback per step via callback / callback_steps)
        step_target = int(pipeline_kwargs['num_inference_steps'])
        phase_iter = iter(default_phases)
        current_phase = next(phase_iter, None)

        def diffusion_callback(step_idx: int, timestep: int, latents):  # noqa: D401
            nonlocal current_phase
            # Cancellation fast-path: check flag early to minimize wasted work
            with self.progress_lock:
                if generation_id in self._cancellation_flags:
                    # Mark canceled if not already terminal
                    existing = self.generation_progress.get(generation_id)
                    if existing and existing.get('status') not in ['canceled', 'error', 'completed']:
                        self.update_progress(
                            generation_id,
                            progress=existing.get('progress', 0.0),
                            status='canceled',
                            message=f'Canceled at step {step_idx+1}/{step_target}',
                            current_step=step_idx+1,
                            total_steps=step_target
                        )
                    raise CancelledGeneration(f"Generation {generation_id} canceled")
            now = time.monotonic()
            if step_idx > 0:
                step_durations.append(now - diffusion_callback.last_time)
            diffusion_callback.last_time = now

            # Compute basic progress proportion for diffusion window (allocate 40% -> 75%)
            diffusion_start_pct = 40.0
            diffusion_end_pct = 75.0
            frac = (step_idx + 1) / step_target
            diffusion_pct = diffusion_start_pct + (diffusion_end_pct - diffusion_start_pct) * frac

            elapsed = now - start_monotonic
            spi = None
            eta_sec = None
            if len(step_durations) >= 3:
                spi = sum(step_durations) / len(step_durations)
                remaining = max(0, step_target - (step_idx + 1))
                eta_sec = remaining * spi

            # Advance phases if needed
            while current_phase and diffusion_pct >= current_phase[0]:
                phase_pct, phase_msg = current_phase
                self.update_progress(generation_id, progress=phase_pct, message=phase_msg, current_step=step_idx+1, total_steps=step_target)
                current_phase = next(phase_iter, None)

            self.update_progress(
                generation_id,
                progress=diffusion_pct,
                message=f"Diffusion step {step_idx+1}/{step_target}",
                current_step=step_idx+1,
                total_steps=step_target,
                elapsed_seconds=elapsed,
                seconds_per_it=spi,
                eta_seconds=eta_sec
            )

        diffusion_callback.last_time = time.monotonic()

        try:
            result = self.pipeline(callback=diffusion_callback, callback_steps=1, **pipeline_kwargs)
            final_img = result.images[0]
        except CancelledGeneration:
            # Propagate after ensuring LoRA unload in finally; caller will handle
            raise
        finally:
            if lora_model and lora_model.lower() not in ['none', ''] and hasattr(self.pipeline, 'unload_lora_weights'):
                try:
                    self.pipeline.unload_lora_weights()
                except Exception:
                    pass

        # Post-diffusion phase updates (allocate remaining headroom to 98% before final 100%)
        try:
            import time
            end_elapsed = time.monotonic() - start_monotonic
            # Emit sequential phase transitions with minimal delay to reflect pipeline post-processing
            self.update_progress(generation_id, progress=80.0, message='Refining latent image', elapsed_seconds=end_elapsed, current_step=step_target, total_steps=step_target)
            self.update_progress(generation_id, progress=90.0, message='Decoding & post-processing', elapsed_seconds=end_elapsed, current_step=step_target, total_steps=step_target)
            self.update_progress(generation_id, progress=95.0, message='Finalizing outputs', elapsed_seconds=end_elapsed, current_step=step_target, total_steps=step_target)
            self.update_progress(generation_id, progress=98.0, message='Preparing result payload', elapsed_seconds=end_elapsed, current_step=step_target, total_steps=step_target)
        except Exception:
            pass  # Non-critical
        # Caller (/generate) will set 100% upon completion of pixel art processing.
        return final_img, seed

    @staticmethod
    def classify_exception(ex: Exception):
        """Return (code, friendly_message) for known generation errors."""
        msg = str(ex) if ex else ""
        lower = msg.lower()
        # Ordered checks from specific to generic
        if msg == "INCOMPATIBLE_LORA" or "size mismatch" in lower or "target modules" in lower:
            return ("INCOMPATIBLE_LORA", "Selected LoRA is incompatible with the current base model. Try a different style or switch the base model.")
        if "no base model loaded" in lower:
            return ("NO_MODEL_LOADED", "Load a base model first (Model Settings > Base Model).")
        if "mlx_not_installed" in lower:
            return ("MLX_NOT_INSTALLED", "MLX not installed. Run: pip install mlx mlx-lm (Apple Silicon only).")
        if "mlx_not_implemented" in lower:
            return ("MLX_NOT_IMPLEMENTED", "MLX backend not fully implemented yet.")
        if "no prompt provided" in lower:
            return ("VALIDATION_ERROR", "Please enter a prompt before generating.")
        if "out of memory" in lower or "cuda" in lower and "memory" in lower:
            return ("OUT_OF_MEMORY", "Out of memory. Use Fast (512x512), fewer steps, or close other apps.")
        if "connection error" in lower or "failed to load model" in lower:
            return ("MODEL_LOAD_ERROR", "Model load failed. Check internet/offline mode and try again.")
        return ("UNKNOWN_ERROR", "Generation failed. See server log for details.")

    def generate_image_mlx(self, prompt, **kwargs):
        """Stub MLX generation pathway. Returns placeholder or raises friendly errors."""
        try:
            try:
                import mlx.core as mx  # noqa: F401
            except ModuleNotFoundError:
                raise Exception("MLX_NOT_INSTALLED")

            # Placeholder image generation using PIL only (fast, offline)
            width = int(kwargs.get('width', 512))
            height = int(kwargs.get('height', 512))
            steps = kwargs.get('num_inference_steps', 30)

            # Simple deterministic placeholder based on prompt hash
            import hashlib, math
            h = int(hashlib.sha1(prompt.encode('utf-8')).hexdigest(), 16)
            base_color = (
                (h >> 0) & 0xFF,
                (h >> 8) & 0xFF,
                (h >> 16) & 0xFF,
                255
            )
            img = Image.new('RGBA', (width, height), base_color)

            # Add simple pattern so different prompts look distinct
            pixels = img.load()
            for y in range(0, height, max(1, height // 32)):
                for x in range(0, width, max(1, width // 32)):
                    if (x // max(1, width // 32) + y // max(1, height // 32)) % 2 == 0:
                        r = (base_color[0] + (x * 7) + (y * 3)) % 255
                        g = (base_color[1] + (x * 5) + (y * 11)) % 255
                        b = (base_color[2] + (x * 13) + (y * 17)) % 255
                        for yy in range(y, min(height, y + max(1, height // 32))):
                            for xx in range(x, min(width, x + max(1, width // 32))):
                                pixels[xx, yy] = (r, g, b, 255)

            # Return placeholder image and synthetic seed (steps influences variability slightly)
            seed = (h ^ (steps << 8)) & 0xFFFFFFFF
            print("🧪 Generated placeholder image via MLX backend stub.")
            return img, seed
        except Exception:
            raise

    def set_engine(self, engine: str):
        """Switch inference engine. 'torch' for diffusers, 'mlx' for Apple MLX backend."""
        engine = (engine or '').lower()
        if engine not in ['torch', 'mlx']:
            raise ValueError("Invalid engine. Use 'torch' or 'mlx'.")
        if engine == self.current_engine:
            return False  # No change
        self.current_engine = engine
        print(f"🔀 Switched inference engine to: {engine}")
        return True

    # ---------------------- Progress Tracking Core (Tasks 1.2.x) ----------------------
    def update_progress(self, generation_id: str, **fields):
        """Thread-safe progress update with monotonic enforcement.

        fields can include: progress(float 0-100), status, message, current_step,
        total_steps, elapsed_seconds, eta_seconds, seconds_per_it, width, height, steps.
        """
        now_dt = datetime.utcnow()
        iso_now = now_dt.replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')
        with self.progress_lock:
            entry = self.generation_progress.get(generation_id)
            if not entry:
                # Initialize entry if absent
                entry = {
                    'generation_id': generation_id,
                    'progress': 0.0,
                    'status': fields.get('status', 'generating'),
                    'message': fields.get('message', ''),
                    'timestamp': iso_now,
                    'created_at': now_dt,
                    'updated_at': now_dt,
                    'current_step': None,
                    'total_steps': None,
                    'elapsed_seconds': None,
                    'eta_seconds': None,
                    'seconds_per_it': None,
                    'width': fields.get('width'),
                    'height': fields.get('height'),
                    'steps': fields.get('steps'),
                    'settings_included': bool(fields.get('width') and fields.get('height') and fields.get('steps'))
                }
                self.generation_progress[generation_id] = entry

            old_progress = entry.get('progress', 0.0)
            new_progress = fields.get('progress', old_progress)
            if self.enforce_monotonic_progress and new_progress < old_progress:
                new_progress = old_progress

            # Merge allowed scalar fields
            scalar_keys = [
                'status','message','current_step','total_steps','elapsed_seconds',
                'eta_seconds','seconds_per_it','width','height','steps'
            ]
            for k in scalar_keys:
                if k in fields and fields[k] is not None:
                    entry[k] = fields[k]

            entry['progress'] = float(new_progress)
            entry['timestamp'] = iso_now
            entry['updated_at'] = now_dt

            # Terminal status expiry setup
            if 'status' in fields and fields['status'] in ('completed','error','canceled') and not entry.get('expires_at'):
                retention = self.progress_retention_seconds
                if fields['status'] == 'error':
                    retention = self.progress_error_retention_seconds
                exp_dt = now_dt + timedelta(seconds=retention)
                entry['expires_at'] = exp_dt.replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z')
                entry['expiry_grace_seconds'] = self.expiry_grace_seconds

            # Logging thresholds
            log_reason = False
            if abs(new_progress - old_progress) >= 1.0:
                log_reason = True
            if 'status' in fields and fields['status'] != entry.get('status'):
                log_reason = True
            if 'message' in fields and fields['message']:
                # log important milestone messages (diffusion steps, phase changes)
                if any(keyword in fields['message'].lower() for keyword in ['step', 'phase', 'complete', 'error', 'canceled']):
                    log_reason = True

            if log_reason:
                print(f"📊 Progress[{generation_id[:8]}]: {entry['progress']:.1f}% status={entry.get('status')} msg={entry.get('message','')[:60]}")

            return entry.copy()

    def get_progress(self, generation_id: str):
        """Retrieve a snapshot of progress data (thread-safe). Returns None if not found."""
        with self.progress_lock:
            entry = self.generation_progress.get(generation_id)
            return None if not entry else {k: v for k, v in entry.items() if k not in ('created_at','updated_at')}

    def cleanup_progress(self):
        """Two-phase expiry:
        1) After retention -> mark status=expired (if not already)
        2) After grace -> remove entry
        """
        now_dt = datetime.utcnow()
        removed = []
        expired_marked = []
        with self.progress_lock:
            for gid, entry in list(self.generation_progress.items()):
                status = entry.get('status')
                updated_at = entry.get('updated_at') or entry.get('created_at') or now_dt
                expires_at_iso = entry.get('expires_at')
                if status in ('completed','error','canceled') and expires_at_iso:
                    # parse expires_at
                    try:
                        from datetime import datetime as _dt
                        expires_at_dt = _dt.fromisoformat(expires_at_iso.replace('Z','+00:00'))
                    except Exception:
                        expires_at_dt = updated_at
                    if now_dt >= expires_at_dt and status != 'expired':
                        entry['status'] = 'expired'
                        entry['timestamp'] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z')
                        entry['expired_at'] = entry['timestamp']
                        expired_marked.append(gid)
                    elif status == 'expired':
                        # purge after grace
                        grace = entry.get('expiry_grace_seconds', self.expiry_grace_seconds)
                        try:
                            expired_at_dt = _dt.fromisoformat(entry.get('expired_at','').replace('Z','+00:00'))
                        except Exception:
                            expired_at_dt = updated_at
                        if now_dt >= expired_at_dt + timedelta(seconds=grace):
                            removed.append(gid)
                            del self.generation_progress[gid]
        if expired_marked:
            print(f"⌛ Marked expired: {', '.join(e[:8] for e in expired_marked)}")
        if removed:
            print(f"🧹 Purged entries: {', '.join(r[:8] for r in removed)}")
        return removed

    def process_for_pixel_art(self, image, target_size=(64, 64), colors=16, preserve_aspect=True, fit_mode="contain", pad_color=(0,0,0,0)):
        """Advanced pixel art post-processing with optional aspect preservation.

        Args:
            image (PIL.Image): High-res diffusion output
            target_size (tuple): (w,h) final pixel art size
            colors (int): Palette size (<=0 disables quantization)
            preserve_aspect (bool): If True, avoid stretching original aspect
            fit_mode (str): One of 'contain', 'cover', 'stretch'
                contain: letterbox/pad so entire image fits inside target
                cover: crop after scaling to completely fill target
                stretch: direct resize (original behavior)
            pad_color (tuple): RGBA used when padding (contain)
        """
        print(f"🖼️ Processing for pixel art: {target_size}, {colors} colors, aspect={preserve_aspect}, fit={fit_mode}")

        tw, th = target_size
        if preserve_aspect and fit_mode in ("contain", "cover"):
            ow, oh = image.width, image.height
            if ow == 0 or oh == 0:
                preserve_aspect = False
            else:
                scale_x = tw / ow
                scale_y = th / oh
                if fit_mode == "contain":
                    scale = min(scale_x, scale_y)
                else:  # cover
                    scale = max(scale_x, scale_y)
                new_w = max(1, int(round(ow * scale)))
                new_h = max(1, int(round(oh * scale)))
                image = image.resize((new_w, new_h), Image.NEAREST)
                # Pad or crop to exact target
                if fit_mode == "contain":
                    canvas = Image.new('RGBA', (tw, th), pad_color)
                    off_x = (tw - new_w) // 2
                    off_y = (th - new_h) // 2
                    canvas.paste(image, (off_x, off_y))
                    image = canvas
                else:  # cover -> crop center
                    if new_w > tw or new_h > th:
                        left = (new_w - tw) // 2
                        top = (new_h - th) // 2
                        image = image.crop((left, top, left + tw, top + th))
                # Ensure final size
                if image.size != (tw, th):
                    image = image.resize((tw, th), Image.NEAREST)
        else:
            # Stretch (original behavior)
            image = image.resize(target_size, Image.NEAREST)
        
        # Apply color quantization if specified
        if colors > 0:
            if image.mode == 'RGBA':
                # Preserve alpha channel during quantization
                alpha = image.getchannel('A')
                rgb_image = image.convert('RGB').quantize(
                    colors=int(colors) - 1,  # Reserve one color for transparency
                    method=Image.MEDIANCUT
                )
                image = rgb_image.convert('RGBA')
                image.putalpha(alpha)
            else:
                image = image.quantize(
                    colors=int(colors),
                    method=Image.MEDIANCUT
                ).convert('RGB')
        
        print("✅ Pixel art processing complete")
        return image

# Initialize the server instance
sd_server = PixelArtSDServer()

@app.route('/generate', methods=['POST'])
def generate():
    """Main generation endpoint with comprehensive error handling."""
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({"success": False, "error": "No prompt provided"}), 400
        
        print(f"\n🎯 New generation request: {prompt[:30]}...")
        # Concurrency guard (single active generation for now)
        with sd_server.progress_lock:
            if sd_server.current_generation_id is not None:
                return jsonify({
                    "success": False,
                    "error": "another generation in progress"
                }), 409
            generation_id = str(uuid4())
            sd_server.current_generation_id = generation_id
        
        # Extract parameters with defaults
        defaults = sd_server.default_settings
        kwargs = {
            "lora_model": data.get('lora_model'),
            "lora_strength": data.get('lora_strength', 1.0),
            "num_inference_steps": data.get('steps', defaults.get('num_inference_steps')),
            "guidance_scale": data.get('guidance_scale', defaults.get('guidance_scale')),
            "seed": data.get('seed', -1),
            "negative_prompt": data.get('negative_prompt', defaults.get('negative_prompt')),
            "width": data.get('width', 1024),   # Base generation resolution
            "height": data.get('height', 1024)  # Base generation resolution
        }
        # Initialize progress entry (0%)
        sd_server.update_progress(
            generation_id,
            progress=0.0,
            status='generating',
            message='Initializing generation',
            width=kwargs['width'],
            height=kwargs['height'],
            steps=kwargs['num_inference_steps']
        )
        
        # Generate the base image
        start_time = datetime.now()
        try:
            if sd_server.current_engine == 'torch':
                image, used_seed = sd_server.generate_image_with_progress(generation_id, prompt=prompt, **kwargs)
            else:
                image, used_seed = sd_server.generate_image(prompt=prompt, **kwargs)
        except CancelledGeneration as cex:
            sd_server.update_progress(
                generation_id,
                status='canceled',
                message=str(cex)[:200]
            )
            with sd_server.progress_lock:
                sd_server.current_generation_id = None
            return jsonify({
                "success": False,
                "error": "generation canceled",
                "code": "CANCELED",
                "generation_id": generation_id
            }), 499  # Client Closed Request semantics (non standard)
        except Exception as gen_ex:
            # Update progress with error
            sd_server.update_progress(
                generation_id,
                status='error',
                message=str(gen_ex)[:200]
            )
            with sd_server.progress_lock:
                sd_server.current_generation_id = None
            raise
        
        # Apply background removal if requested
        if data.get('remove_background', False):
            print("🎭 Applying background removal...")
            image = sd_server.remove_background(image)
        
        # Process for pixel art
        pixel_width = int(data.get('pixel_width', 64))
        pixel_height = int(data.get('pixel_height', 64))
        colors = int(data.get('colors', 16))
        
        preserve_aspect = bool(data.get('preserve_aspect', True))
        fit_mode = data.get('fit_mode', 'contain')
        if fit_mode not in ['contain', 'cover', 'stretch']:
            fit_mode = 'contain'

        pixel_image = sd_server.process_for_pixel_art(
            image,
            target_size=(pixel_width, pixel_height),
            colors=colors,
            preserve_aspect=preserve_aspect,
            fit_mode=fit_mode
        )
        
        # Convert to base64
        img_base64 = sd_server.image_to_base64(pixel_image)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️ Total generation time: {generation_time:.2f}s")
        # Update progress to completed (100%)
        sd_server.update_progress(
            generation_id,
            progress=100.0,
            status='completed',
            message='Generation complete'
        )
        # Schedule cleanup of current_generation_id and progress retention
        def _post_cleanup():
            import time
            # Quick release to allow a new generation after short grace
            time.sleep(1)
            with sd_server.progress_lock:
                sd_server.current_generation_id = None
            # Wait remaining retention period before marking expired automatically
            remaining = max(0, sd_server.progress_retention_seconds - 1)
            if remaining:
                time.sleep(remaining)
            sd_server.cleanup_progress()
        threading.Thread(target=_post_cleanup, daemon=True).start()
        
        return jsonify({
            "success": True,
            "image": {
                "base64": img_base64,
                "width": pixel_width,
                "height": pixel_height,
                "mode": "rgba",
                "fit_mode": fit_mode,
                "preserve_aspect": preserve_aspect
            },
            "seed": used_seed,
            "prompt": prompt,
            "generation_time": generation_time,
            "generation_id": generation_id
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        code, friendly = sd_server.classify_exception(e)
        print(f"❌ Generation error [{code}]: {friendly}")
        # Ensure guard released on errors
        with sd_server.progress_lock:
            # Attempt to mark active generation as errored
            active_id = sd_server.current_generation_id
            if active_id:
                try:
                    sd_server.update_progress(active_id, status='error', message=friendly, progress=100.0 if code == 'OUT_OF_MEMORY' else sd_server.generation_progress.get(active_id, {}).get('progress', 0.0))
                except Exception:
                    pass
            sd_server.current_generation_id = None
        return jsonify({
            "success": False,
            "error": friendly,
            "code": code
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Server health and status endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": sd_server.model_loaded,
        "current_model": sd_server.current_model,
        "device": sd_server.device,
    "engine": sd_server.current_engine,
        "version": "2.0.0"
    })

@app.route('/progress/<generation_id>', methods=['GET'])
def progress(generation_id):
    """Progress polling endpoint.

    Returns 200 with progress JSON, 400 for invalid UUID, 404 if not found, 410 if expired.
    """
    # UUID validation
    try:
        UUID(generation_id)
    except Exception:
        return jsonify({"success": False, "error": "invalid id"}), 400

    # Cleanup expired entries opportunistically
    sd_server.cleanup_progress()

    entry = sd_server.get_progress(generation_id)
    if not entry:
        # Could be expired or never existed; we can't fully distinguish without separate tombstone tracking, return 404
        return jsonify({"success": False, "error": "not found"}), 404

    status = entry.get('status')
    # Build response mapping
    resp = {
        "success": True,
        "generation_id": entry.get('generation_id'),
        "progress": entry.get('progress'),
        "status": status,
        "message": entry.get('message'),
        "timestamp": entry.get('timestamp'),
        "current_step": entry.get('current_step'),
        "total_steps": entry.get('total_steps'),
        "elapsed_seconds": entry.get('elapsed_seconds'),
        "eta_seconds": entry.get('eta_seconds'),
        "seconds_per_it": entry.get('seconds_per_it'),
        "width": entry.get('width'),
        "height": entry.get('height'),
        "steps": entry.get('steps'),
        "expires_at": entry.get('expires_at'),
        "expired_at": entry.get('expired_at')
    }

    # Future: explicit expired tracking -> 410
    if status == 'expired':
        return jsonify({"success": False, "generation_id": entry.get('generation_id'), "status": "expired"}), 410

    return jsonify(resp)

@app.route('/cancel/<generation_id>', methods=['POST'])
def cancel_generation(generation_id):
    """Cancellation endpoint.

    Sets a cancellation flag inspected inside the diffusion callback. Returns:
    202 if cancellation accepted and generation in-progress
    404 if no such generation exists (never existed or already cleaned)
    409 if generation already terminal (completed / error / canceled)
    400 if invalid UUID
    """
    # Validate UUID
    try:
        UUID(generation_id)
    except Exception:
        return jsonify({"success": False, "error": "invalid id"}), 400

    with sd_server.progress_lock:
        entry = sd_server.generation_progress.get(generation_id)
        if not entry:
            return jsonify({"success": False, "error": "not found"}), 404
        status = entry.get('status')
        if status in ['completed', 'error', 'canceled', 'expired']:
            return jsonify({"success": False, "error": f'already {status}'}), 409
        # Set cancellation flag
        sd_server._cancellation_flags.add(generation_id)
        # Optional immediate progress update message (don't change status yet to avoid prematurely terminal state)
        sd_server.update_progress(generation_id, message='Cancellation requested')
        return jsonify({
            "success": True,
            "generation_id": generation_id,
            "status": 'canceling'
        }), 202

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """Load a specific AI model."""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({"success": False, "error": "No model_name provided"}), 400
        
        print(f"📦 Loading model: {model_name}")
        
        if sd_server.load_model(model_name):
            return jsonify({
                "success": True,
                "model": model_name,
                "device": sd_server.device
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Failed to load {model_name}"
            }), 500
            
    except Exception as e:
        print(f"❌ Model loading error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/set_engine', methods=['POST'])
def set_engine_route():
    """Switch inference engine between 'torch' and 'mlx'."""
    try:
        data = request.get_json() or {}
        engine = data.get('engine')
        if not engine:
            return jsonify({"success": False, "error": "No engine provided"}), 400
        changed = sd_server.set_engine(engine)
        return jsonify({
            "success": True,
            "engine": sd_server.current_engine,
            "changed": changed
        })
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available base models."""
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5"
    ]
    return jsonify({"models": models})

@app.route('/loras', methods=['GET'])
def list_loras():
    """List available LoRA models (both hub and local)."""
    base_list = [
        "None",
        # SDXL LoRAs (examples)
        "nerijs/pixel-art-xl",
        "ntc-ai/SDXL-LoRA-slider.pixel-art"
    ]

    # Add local LoRA files
    lora_directory = "loras"
    if not os.path.isdir(lora_directory):
        os.makedirs(lora_directory)

    for filename in os.listdir(lora_directory):
        if filename.endswith(".safetensors"):
            lora_path = os.path.join(lora_directory, filename).replace("\\", "/")
            if lora_path not in base_list:
                base_list.append(lora_path)

    all_loras = list(base_list)
    compatible_loras = list(all_loras)
    incompatible_loras = []

    # Determine compatibility but do NOT hide incompatible ones; just tag them
    try:
        if sd_server.model_loaded and sd_server.current_model:
            is_xl = 'xl' in sd_server.current_model.lower()
            compatible_loras = [m for m in all_loras if m == 'None' or (("xl" in m.lower()) == is_xl)]
            incompatible_loras = [m for m in all_loras if m not in compatible_loras and m != 'None']
    except Exception as e:
        print(f"⚠️ LoRA compatibility evaluation error: {e}")

    return jsonify({
        "loras": all_loras,
        "compatible_loras": compatible_loras,
        "incompatible_loras": incompatible_loras
    })

def main(default_model_to_load=None, offline=False):
    """Main server startup function."""
    print("\n" + "="*50)
    print("🎮 LOCAL AI GENERATOR SERVER v2.0")
    print("="*50)
    
    sd_server.offline_mode = offline
    
    if default_model_to_load and default_model_to_load.lower() != "none":
        print(f"📦 Loading default model: {default_model_to_load}")
        sd_server.load_model(default_model_to_load)
    else:
        print("⏸️ No model loaded on startup (will load on first request)")
    
    print("\n🌐 Server Configuration:")
    print(f"   • Host: 127.0.0.1")
    print(f"   • Port: 5000")
    print(f"   • Device: {sd_server.device}")
    print(f"   • Offline Mode: {offline}")
    
    print("\n✅ Server ready! Access at http://127.0.0.1:5000")
    print("🎨 Aseprite plugin can now connect and generate images!")
    print("\n" + "="*50 + "\n")
    
    try:
        # Disable Flask development server warning
        cli = sys.modules['flask.cli']
        cli.show_server_banner = lambda *x: None
        
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Server shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main(default_model_to_load="stabilityai/stable-diffusion-xl-base-1.0")