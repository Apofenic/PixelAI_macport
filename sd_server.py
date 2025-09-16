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
from datetime import datetime
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
        self.pipeline = None
        self.segmentation_model = None
        self.segmentation_processor = None
        self.model_loaded = False
        self.current_model = None
        # Current inference engine: 'torch' (diffusers) or 'mlx' (Apple MLX backend placeholder)
        self.current_engine = 'torch'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cache = {}
        self.offline_mode = False

        self.default_settings = {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality",
            "pixel_art_prompt_suffix": ", pixel art, 8bit style, game sprite"
        }

        print(f"🚀 Local AI Generator Server v2.0")
        print(f"📱 Device: {self.device}")
        print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

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
        
        # Generate the base image
        start_time = datetime.now()
        image, used_seed = sd_server.generate_image(prompt=prompt, **kwargs)
        
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
            "generation_time": generation_time
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        code, friendly = sd_server.classify_exception(e)
        print(f"❌ Generation error [{code}]: {friendly}")
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