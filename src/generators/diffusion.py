"""
Context-Aware Generative Diffusion Engine (Resolution Fix & 4-Region Mask).

Generates unique, realistic swimmers directly inside RipVIS water scenes using
SDXL / SD 1.5 Inpainting with resolution fix (generate at 300px scale, then downscale to target size).
Includes robust water mask heuristic for sky and sand rejection.
"""

from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

from config import config


def water_mask_simple(image_bgr_or_pil) -> np.ndarray:
    """
    Color-threshold water mask tuned for aerial ocean / rip-current drone footage.
    Robustly rejects sky (blue, gray, bright clouds) and sand (warm red/yellow/brown hues).
    Returns boolean numpy array (H, W).
    """
    if isinstance(image_bgr_or_pil, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(image_bgr_or_pil, cv2.COLOR_BGR2RGB))
    else:
        img_pil = image_bgr_or_pil

    arr = np.array(img_pil.convert("RGB")).astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    h, w = arr.shape[:2]

    hsv = np.array(img_pil.convert("HSV")).astype(np.int16)
    hue, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    is_sand = (
        ((r > b + 2) & (r > g - 12)) |
        ((hue >= 5) & (hue <= 40) & (sat > 20) & (v > 60)) |
        (r > g + 10)
    )
    is_land_veg = (g > b + 25) & (g > r + 15)

    is_blue_sky = (b > g + 8) & (b > r + 15) & (v > 140)
    is_gray_sky = (sat < 45) & (v > 150)
    is_too_bright = (v > 220) | ((r + g + b) > 650)
    is_sky_color = is_blue_sky | is_gray_sky | is_too_bright

    is_waterish = (
        (b >= r - 5) &
        (g >= r - 5) &
        ((b + g) > (2 * r - 5)) &
        (v >= 20)
    )

    mask = is_waterish & (~is_sand) & (~is_land_veg) & (~is_sky_color)

    top_40_cutoff = int(h * 0.40)
    if top_40_cutoff > 0:
        top_sky_or_haze = (b[0:top_40_cutoff, :] <= r[0:top_40_cutoff, :] + 5) | (v[0:top_40_cutoff, :] > 160)
        mask[0:top_40_cutoff, :][top_sky_or_haze] = False

    return mask


def build_4region_mask(
    canvas_size: int = 1024,
    center_xy: tuple[int, int] = (512, 512),
    target_person_height: int = 300,
    sun_azimuth_deg: float = 45.0,
) -> dict[str, Any]:
    """
    Build 4-region mask decomposition centered at center_xy in canvas_size image.
    Decomposes into Body, Submerged Smear, Wake Ring, Shadow.
    """
    cx, cy = center_xy
    bw = int(target_person_height * 0.6)
    bh = target_person_height

    body = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.ellipse(body, (cx, cy), (max(4, bw // 2), max(6, bh // 2)), 0, 0, 360, 255, -1)

    smear_shift = max(3, int(bh * 1.5))
    kernel_smear = np.zeros((smear_shift, 3), dtype=np.uint8)
    kernel_smear[:, 1] = 1
    smear = cv2.threshold(cv2.dilate(body, kernel_smear), 50, 255, cv2.THRESH_BINARY)[1]

    gt_mask = cv2.bitwise_or(body, smear)
    gt_cnts, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_bbox = cv2.boundingRect(np.vstack(gt_cnts)) if gt_cnts else (cx - bw // 2, cy - bh // 2, bw, bh)

    wake_k = max(3, int(bw * 0.5) // 2 * 2 + 1)
    wake_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wake_k, wake_k))
    wake = cv2.subtract(cv2.dilate(body, wake_kernel), body)

    rad = np.radians(sun_azimuth_deg)
    dx, dy = int(np.cos(rad) * bh * 0.4), int(np.sin(rad) * bh * 0.4)
    shadow = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.ellipse(shadow, (cx + dx, cy + bh // 2 + dy), (max(4, int(bw * 0.7)), max(3, int(bh * 0.3))), sun_azimuth_deg, 0, 360, 255, -1)

    union = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(body, smear), wake), shadow)

    return {
        "body": body,
        "smear": smear,
        "wake": wake,
        "shadow": shadow,
        "gt_mask": gt_mask,
        "gt_bbox": gt_bbox,
        "union": union,
    }


def crop_and_feather_patch(inpainted_bgr: np.ndarray, bg_canvas_bgr: np.ndarray, union_mask: np.ndarray, feather_radius: int = 16) -> np.ndarray:
    inp_arr = inpainted_bgr.astype(np.float32)
    bg_arr = bg_canvas_bgr.astype(np.float32)

    border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated_mask = cv2.dilate(union_mask.astype(np.uint8), border_kernel) > 0
    surround_water = dilated_mask & (union_mask == 0)

    if np.any(surround_water):
        bg_mean = bg_arr[surround_water].mean(axis=0)
        bg_std = bg_arr[surround_water].std(axis=0) + 1e-5
        inp_mean = inp_arr[union_mask > 0].mean(axis=0)
        inp_std = inp_arr[union_mask > 0].std(axis=0) + 1e-5

        inp_matched = (inp_arr - inp_mean) * (bg_std / inp_std) + bg_mean
        inp_arr = np.clip(inp_matched, 0, 255)

    mask_pil = Image.fromarray(union_mask)
    feathered_mask = mask_pil.filter(ImageFilter.GaussianBlur(feather_radius))
    alpha = np.array(feathered_mask).astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]

    blended = inp_arr * alpha + bg_arr * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


class DiffusionPersonGenerator:
    """Generative Diffusion model synthesizing unique people directly into RipVIS ocean scenes via IP-Adapter visual crop conditioning."""

    def __init__(self, device: str = config.DEVICE) -> None:
        self.device = device
        self.pipe = None
        if HAS_DIFFUSERS and torch.cuda.is_available() and device != "cpu":
            try:
                print("[DiffusionPersonGenerator] Loading SDXL Inpainting Pipeline...")
                self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )
                print("[DiffusionPersonGenerator] Loading SDXL IP-Adapter weights (h94/IP-Adapter)...")
                self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors")
                self.pipe.set_ip_adapter_scale(0.8)  # Working baseline IP-Adapter scale
                self.pipe.enable_model_cpu_offload()
                self.pipe.vae.enable_slicing()
                self.pipe.vae.enable_tiling()
                print("[DiffusionPersonGenerator] SDXL Inpainting + IP-Adapter loaded successfully!")
            except Exception as exc:
                print(f"[Notice] SDXL IP-Adapter Load Error: ({exc}).")

    def generate_person_in_water(
        self,
        bg_img: np.ndarray,
        placement_xy: tuple[int, int],
        reference_crop: Any = None,
        target_person_height: int = 40,
        large_render_size: int = 300,
        canvas_size: int = 1024,
        strength: float = 0.98,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        bg_h, bg_w = bg_img.shape[:2]
        px, py = placement_xy
        half_sz = canvas_size // 2

        x1, y1 = max(0, px - half_sz), max(0, py - half_sz)
        x2, y2 = min(bg_w, px + half_sz), min(bg_h, py + half_sz)

        bg_patch = bg_img[y1:y2, x1:x2]
        pad_t, pad_b = max(0, -(py - half_sz)), max(0, (py + half_sz) - bg_h)
        pad_l, pad_r = max(0, -(px - half_sz)), max(0, (px + half_sz) - bg_w)

        canvas = cv2.copyMakeBorder(bg_patch, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        canvas = cv2.resize(canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_LANCZOS4)

        mask_decomp = build_4region_mask(
            canvas_size=canvas_size,
            center_xy=(canvas_size // 2, canvas_size // 2),
            target_person_height=large_render_size,
        )
        union_mask = mask_decomp["union"]
        gt_bbox_1024 = mask_decomp["gt_bbox"]

        if self.pipe is not None and reference_crop is not None:
            img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(union_mask)

            if isinstance(reference_crop, np.ndarray):
                ref_pil = Image.fromarray(cv2.cvtColor(reference_crop, cv2.COLOR_BGR2RGB))
            else:
                ref_pil = reference_crop

            res_pil = self.pipe(
                prompt="a swimmer in ocean water, aerial view, realistic drone photo",
                negative_prompt="blurry, land, beach, sand, boat",
                image=img_pil,
                mask_image=mask_pil,
                ip_adapter_image=ref_pil,
                strength=0.98,
                guidance_scale=6.0,
                num_inference_steps=20,
            ).images[0]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            inp_bgr = cv2.cvtColor(np.array(res_pil), cv2.COLOR_RGB2BGR)

            # Crop tightly around generated region and feather alpha edge
            ys, xs = np.where(union_mask > 0)
            if len(ys) > 0:
                pad_px = 20
                min_x, max_x = max(0, int(xs.min() - pad_px)), min(canvas_size, int(xs.max() + pad_px))
                min_y, max_y = max(0, int(ys.min() - pad_px)), min(canvas_size, int(ys.max() + pad_px))

                crop_rgb = inp_bgr[min_y:max_y, min_x:max_x]
                mask_crop = union_mask[min_y:max_y, min_x:max_x]

                scale = target_person_height / float(large_render_size)
                down_w = max(1, int(crop_rgb.shape[1] * scale))
                down_h = max(1, int(crop_rgb.shape[0] * scale))

                crop_rgb_down = cv2.resize(crop_rgb, (down_w, down_h), interpolation=cv2.INTER_AREA)
                mask_crop_down = cv2.resize(mask_crop, (down_w, down_h), interpolation=cv2.INTER_NEAREST)
                crop_alpha_down = (mask_crop_down > 0).astype(np.float32)[..., None]

                # Match capture noise
                noise = np.random.normal(0, 3, crop_rgb_down.shape).astype(np.float32)
                crop_rgb_down = np.clip(crop_rgb_down.astype(np.float32) + noise, 0, 255).astype(np.uint8)

                out_bg = bg_img.copy()
                p_x1 = max(0, int(px - down_w / 2))
                p_y1 = max(0, int(py - down_h / 2))
                p_x2 = min(bg_w, p_x1 + down_w)
                p_y2 = min(bg_h, p_y1 + down_h)

                crop_h, crop_w = p_y2 - p_y1, p_x2 - p_x1
                if crop_h > 0 and crop_w > 0:
                    patch_rgb = crop_rgb_down[:crop_h, :crop_w].astype(np.float32)
                    patch_alpha = crop_alpha_down[:crop_h, :crop_w].astype(np.float32)
                    bg_crop = out_bg[p_y1:p_y2, p_x1:p_x2].astype(np.float32)

                    blended = patch_rgb * patch_alpha + bg_crop * (1.0 - patch_alpha)
                    out_bg[p_y1:p_y2, p_x1:p_x2] = np.clip(blended, 0, 255).astype(np.uint8)

                box_w = int(target_person_height * 0.55)
                box_h = int(target_person_height * 1.4)

                # Return 384x384 patch centered on swimmer for high-resolution YOLO training
                patch_sz = 384
                half_p = patch_sz // 2
                crop_x1, crop_y1 = max(0, px - half_p), max(0, py - half_p)
                crop_x2, crop_y2 = min(bg_w, crop_x1 + patch_sz), min(bg_h, crop_y1 + patch_sz)

                out_patch = out_bg[crop_y1:crop_y2, crop_x1:crop_x2]
                rel_bx = max(0, px - crop_x1 - box_w // 2)
                rel_by = max(0, py - crop_y1 - box_h // 2)
                rel_gt_bbox = (rel_bx, rel_by, box_w, box_h)
                return out_patch, rel_gt_bbox

            return bg_img.copy()[:384, :384], (180, 180, 20, 20)
        else:
            return bg_img.copy()[:384, :384], (180, 180, 20, 20)
