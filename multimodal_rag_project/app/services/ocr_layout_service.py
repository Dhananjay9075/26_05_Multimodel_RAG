# title="app/services/ocr_layout_service.py"
import os
import gc  # Force memory reclamation
import sys
import re
import glob
import numpy as np
from PIL import Image
from typing import Dict, Any, List
from app.config import settings

# Lazy initialization placeholders for heavyweight deep learning architectures
_paddle_ocr = None
_yolo_layout = None
_florence_model = None
_florence_processor = None

def get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    return _paddle_ocr

def get_yolo_layout():
    global _yolo_layout
    if _yolo_layout is None:
        from ultralytics import YOLO
        try:
            _yolo_layout = YOLO("yolov8x-cls.pt")
        except Exception:
            _yolo_layout = None
    return _yolo_layout


# ---------------------------------------------------------------------------
# HuggingFace cache helpers
# ---------------------------------------------------------------------------

def _find_hf_cache_roots():
    """Return all existing HuggingFace cache root directories."""
    user_home = os.path.expanduser("~")
    roots = [
        os.environ.get("HF_HOME", ""),
        os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
        os.path.join(user_home, ".cache", "huggingface"),
        os.path.join(user_home, ".cache", "huggingface", "hub"),
        # Windows-specific
        os.path.join(os.environ.get("USERPROFILE", ""), ".cache", "huggingface"),
        os.path.join(os.environ.get("APPDATA", ""), "huggingface"),
    ]
    return [r for r in roots if r and os.path.isdir(r)]


def _patch_file_on_disk(filename, sentinel, find_str, replace_str, description,
                        bad_patch_str=None):
    """
    Generic helper: find all copies of `filename` under the HF cache,
    and replace `find_str` with `replace_str` if `sentinel` is not already present.

    If `bad_patch_str` is given, files containing it are treated as incorrectly
    patched and will be re-patched (the bad string is replaced with replace_str).

    Returns True if at least one file was (or had already been) correctly patched.
    """
    found_files = []
    for root in _find_hf_cache_roots():
        found_files.extend(glob.glob(
            os.path.join(root, "**", filename), recursive=True
        ))

    if not found_files:
        print(f"⚠️  Could not find cached {filename} to patch ({description}).")
        return False

    patched_count = 0
    for fpath in found_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                source = f.read()

            # Detect and fix a previously-applied bad patch
            if bad_patch_str and bad_patch_str in source:
                patched = source.replace(bad_patch_str, replace_str)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(patched)
                print(f"✅ Re-patched (fixed bad patch) {filename}: {fpath}")
                patched_count += 1
                continue

            if sentinel in source:          # already correctly patched
                patched_count += 1
                continue

            if find_str not in source:      # different version, nothing to fix
                continue

            patched = source.replace(find_str, replace_str)
            if patched == source:
                print(f"⚠️  Replacement had no effect on {fpath}")
                continue

            with open(fpath, "w", encoding="utf-8") as f:
                f.write(patched)
            print(f"✅ Patched {filename}: {fpath}")
            patched_count += 1

        except Exception as e:
            print(f"⚠️  Failed to patch {fpath}: {e}")

    return patched_count > 0


def _patch_florence_config_file():
    """
    FIX 1 — 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'

    transformers >= 4.50 changed PreTrainedConfig.__getattribute__ to raise
    AttributeError for missing attributes. Florence2LanguageConfig.__init__
    reads self.forced_bos_token_id BEFORE calling super().__init__, so it
    crashes. Fix: inject it as a class-level attribute directly in the source.
    """
    SENTINEL    = "# PATCHED_forced_bos_token_id"
    FIND        = "class Florence2LanguageConfig("
    REPLACE     = f"class Florence2LanguageConfig(\n    forced_bos_token_id = None  {SENTINEL}"

    # The regex approach is more reliable — insert one line after the class header
    found_files = []
    for root in _find_hf_cache_roots():
        found_files.extend(glob.glob(
            os.path.join(root, "**", "configuration_florence2.py"), recursive=True
        ))

    if not found_files:
        print("⚠️  Could not find cached configuration_florence2.py to patch.")
        return False

    patched_count = 0
    for fpath in found_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                source = f.read()

            if SENTINEL in source:
                patched_count += 1
                continue

            if "forced_bos_token_id" not in source and \
               "Florence2LanguageConfig" not in source:
                continue

            # Insert class-level attr right after the class ... : line
            fix_line = f"    forced_bos_token_id = None  {SENTINEL}\n"
            patched = re.sub(
                r'(class Florence2LanguageConfig\b[^\n]*\n)',
                r'\1' + fix_line,
                source,
                count=1
            )

            if patched == source:
                print(f"⚠️  Config patch had no effect on {fpath}")
                continue

            with open(fpath, "w", encoding="utf-8") as f:
                f.write(patched)
            print(f"✅ Patched Florence-2 config file: {fpath}")
            patched_count += 1

        except Exception as e:
            print(f"⚠️  Failed to patch config {fpath}: {e}")

    return patched_count > 0


def _patch_florence_processing_file():
    """
    FIX 2 — 'RobertaTokenizer' has no attribute 'additional_special_tokens'

    transformers >= 5.0 removed `additional_special_tokens` as a direct
    attribute on PreTrainedTokenizer. Florence-2's processing_florence2.py
    does:
        tokenizer.additional_special_tokens + \\
            ['<od>', ...]
    The attribute access raises AttributeError. Fix: replace with getattr().

    IMPORTANT: The original token is on a line ending with `+ \\` (line
    continuation). We must NOT append an inline comment to the replacement
    because that would comment out the `+ \\`, breaking the list literal.
    The sentinel is placed on a preceding standalone comment line instead.
    """
    SENTINEL = "# PATCHED_additional_special_tokens"
    FIND     = "tokenizer.additional_special_tokens"
    # Replace just the attribute access; leave the rest of the line (+ \...) intact.
    # Sentinel goes on a new line BEFORE the replacement so it never interrupts syntax.
    REPLACE  = f"{SENTINEL}\n            getattr(tokenizer, 'additional_special_tokens', [])"

    # The previous (bad) patch appended the sentinel inline on the same line,
    # which commented out the `+ \` continuation and broke the list literal.
    BAD_PATCH = f"getattr(tokenizer, 'additional_special_tokens', [])  {SENTINEL}"

    return _patch_file_on_disk(
        filename="processing_florence2.py",
        sentinel=SENTINEL,
        find_str=FIND,
        replace_str=REPLACE,
        description="RobertaTokenizer.additional_special_tokens",
        bad_patch_str=BAD_PATCH,
    )

def _patch_florence_modeling_file():
    """
    FIX 3 — 'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'

    transformers >= 4.54 added a check in PreTrainedModel.__init__ that reads
    self._supports_sdpa. Florence-2's custom Florence2ForConditionalGeneration
    class does not define this attribute, so super().__init__(config) crashes.

    Fix: add `_supports_sdpa = False` as a class-level attribute in the cached
    modeling_florence2.py so the attribute exists when __init__ runs.
    The class also needs _supports_flash_attn_2 = False for the same reason.
    """
    SENTINEL = "# PATCHED_supports_sdpa"

    found_files = []
    for root in _find_hf_cache_roots():
        found_files.extend(glob.glob(
            os.path.join(root, "**", "modeling_florence2.py"), recursive=True
        ))

    if not found_files:
        print("\u26a0\ufe0f  Could not find cached modeling_florence2.py to patch.")
        return False

    patched_count = 0
    for fpath in found_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                source = f.read()

            if SENTINEL in source:
                patched_count += 1
                continue

            if "Florence2ForConditionalGeneration" not in source:
                continue

            fix_lines = (
                f"    _supports_sdpa = False  {SENTINEL}\n"
                f"    _supports_flash_attn_2 = False  {SENTINEL}\n"
            )
            patched = re.sub(
                r"(class Florence2ForConditionalGeneration\b[^\n]*\n)",
                r"\1" + fix_lines,
                source,
                count=1
            )

            if patched == source:
                print(f"\u26a0\ufe0f  Modeling patch had no effect on {fpath}")
                continue

            with open(fpath, "w", encoding="utf-8") as f:
                f.write(patched)
            print(f"\u2705 Patched Florence-2 modeling file: {fpath}")
            patched_count += 1

        except Exception as e:
            print(f"\u26a0\ufe0f  Failed to patch modeling {fpath}: {e}")

    return patched_count > 0


def _invalidate_florence_modules():
    """Drop any already-imported Florence-2 modules so patched files are re-read."""
    to_drop = [
        k for k in sys.modules
        if any(s in k for s in (
            "configuration_florence2", "processing_florence2",
            "modeling_florence2", "Florence2"
        ))
    ]
    for mod in to_drop:
        del sys.modules[mod]


# ---------------------------------------------------------------------------
# Main engine loader
# ---------------------------------------------------------------------------

def get_florence_engine():
    global _florence_model, _florence_processor
    if _florence_model is None:
        import transformers.dynamic_module_utils
        import torch

        # === 1. BYPASS WINDOWS FLASH_ATTN CHECK ===
        if not hasattr(transformers.dynamic_module_utils, '_patched_for_florence'):
            _orig = transformers.dynamic_module_utils.get_imports
            def _no_flash(filename):
                imports = _orig(filename)
                if "flash_attn" in imports:
                    imports.remove("flash_attn")
                return imports
            transformers.dynamic_module_utils.get_imports = _no_flash
            transformers.dynamic_module_utils._patched_for_florence = True

        # === 2. PATCH CACHED SOURCE FILES ON DISK ===
        # Both errors crash inside class constructors before any runtime
        # object patching is possible, so we must fix the source files.
        _patch_florence_config_file()       # fixes: forced_bos_token_id
        _patch_florence_processing_file()   # fixes: additional_special_tokens
        _patch_florence_modeling_file()     # fixes: _supports_sdpa

        # === 3. INVALIDATE ALREADY-IMPORTED MODULES ===
        _invalidate_florence_modules()

        # === 4. LOAD PROCESSOR & MODEL ===
        from transformers import AutoProcessor, AutoModelForCausalLM

        _florence_processor = AutoProcessor.from_pretrained(
            settings.FLORENCE_MODEL,
            trust_remote_code=True
        )

        # Force float32 explicitly to prevent dtype mismatch errors.
        # transformers >= 4.54 may auto-select float16 based on device/config,
        # but Florence-2's remote code does not handle mixed precision cleanly,
        # causing "Input type (float) and bias type (Half) should be the same".
        # We also fix weight tying manually since new transformers reports
        # lm_head / embed_tokens as MISSING (they are tied, not stored separately).
        _florence_model = AutoModelForCausalLM.from_pretrained(
            settings.FLORENCE_MODEL,
            trust_remote_code=True,
            dtype=torch.float32,                # prevent float16 dtype mismatch (torch_dtype deprecated in transformers>=4.54)
            ignore_mismatched_sizes=True,       # tolerate tied-weight key warnings
            _fast_init=False,                   # forces proper weight tying; without this
                                                # lm_head/embed_tokens are randomly inited
        ).eval()

        # === 5. FIX WEIGHT TYING (lm_head / embed_tokens reported as MISSING) ===
        # In Florence-2, lm_head and embed_tokens share weights with the embedding
        # table. New transformers no longer auto-ties them silently. We do it here.
        try:
            lm = _florence_model.language_model
            if hasattr(lm, "lm_head") and hasattr(lm.model, "decoder"):
                lm.lm_head.weight = lm.model.decoder.embed_tokens.weight
            if hasattr(lm.model, "encoder") and hasattr(lm.model, "decoder"):
                lm.model.encoder.embed_tokens.weight = lm.model.decoder.embed_tokens.weight
        except Exception as _tie_err:
            print(f"⚠️  Weight tying skipped: {_tie_err}")

        # === 6. PATCH generation_config (created at runtime, not from file) ===
        for obj in (_florence_model,
                    getattr(_florence_model, "language_model", None)):
            if obj is None:
                continue
            gen_cfg = getattr(obj, "generation_config", None)
            if gen_cfg and not hasattr(gen_cfg, "forced_bos_token_id"):
                gen_cfg.forced_bos_token_id = None

        if torch.cuda.is_available():
            _florence_model = _florence_model.to("cuda")

    return _florence_model, _florence_processor


class OCRLayoutVisionService:
    @staticmethod
    def process_image_elements(image_path: str) -> Dict[str, Any]:
        results = {
            "raw_ocr_text": "",
            "layout_regions": [],
            "visual_caption": "No visual description generated.",
            "structured_analysis": ""
        }

        if not os.path.exists(image_path):
            return results

        # Track tracking variables cleanly to ensure memory dump context
        inputs = None
        inputs_ocr = None
        generated_ids = None
        ocr_ids = None
        image = None

        # === BULLETPROOF FLORENCE-2 ENGINE ===
        try:
            model, processor = get_florence_engine()
            image = Image.open(image_path).convert("RGB")
            import torch

            # Determine device and model dtype so inputs always match
            model_device = next(model.parameters()).device
            model_dtype  = next(model.parameters()).dtype

            def _prepare_inputs(raw_inputs):
                """Move inputs to correct device and cast pixel_values to model dtype."""
                result = {k: v.to(model_device) for k, v in raw_inputs.items()}
                if "pixel_values" in result:
                    result["pixel_values"] = result["pixel_values"].to(model_dtype)
                return result

            # Florence-2's vision encoder asserts square feature maps.
            # New transformers CLIPImageProcessor may produce non-square tensors
            # (e.g. due to padding changes). Pre-resize to the model's expected
            # square crop size (768x768 for Florence-2-base) before processing.
            img_size = getattr(processor.image_processor, "crop_size", None)
            if img_size and isinstance(img_size, dict):
                w = img_size.get("width", img_size.get("height", 768))
                h = img_size.get("height", img_size.get("width", 768))
            else:
                w, h = 768, 768
            image_sq = image.resize((w, h), Image.LANCZOS)

            # 1. Get the Detailed Visual Caption
            inputs = processor(text="<DETAILED_CAPTION>", images=image_sq, return_tensors="pt")
            inputs = _prepare_inputs(inputs)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    use_cache=False,    # transformers>=4.54 wraps past_key_values in
                                        # EncoderDecoderCache; remote code treats it as
                                        # a tuple and crashes with "not subscriptable"
                )
            results["visual_caption"] = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"✅ Florence-2 Caption Success: {results['visual_caption'][:50]}...")

            # 2. Get the OCR Text directly from Florence-2
            inputs_ocr = processor(text="<OCR>", images=image_sq, return_tensors="pt")
            inputs_ocr = _prepare_inputs(inputs_ocr)

            with torch.no_grad():
                ocr_ids = model.generate(
                    input_ids=inputs_ocr["input_ids"],
                    pixel_values=inputs_ocr["pixel_values"],
                    max_new_tokens=1024,
                    use_cache=False,    # same EncoderDecoderCache fix as above
                )
            results["raw_ocr_text"] = processor.batch_decode(ocr_ids, skip_special_tokens=True)[0]
            print("✅ Florence-2 OCR Success!")

        except Exception as e:
            print(f"\n❌ FLORENCE-2 VISION ERROR: {str(e)}\n")
            results["visual_caption"] = f"Vision Engine Error: {str(e)}"
            results["raw_ocr_text"] = "Error extracting OCR data."

        finally:
            # === AGGRESSIVE MEMORY CLEANUP ===
            import torch
            if inputs is not None: del inputs
            if inputs_ocr is not None: del inputs_ocr
            if generated_ids is not None: del generated_ids
            if ocr_ids is not None: del ocr_ids
            if image is not None: del image

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return results