"""Feature extraction methods for player jersey crops."""

from __future__ import annotations

from typing import Callable, Dict, Tuple
import cv2
import numpy as np

from src.classification.cropper import compute_grass_mask


def _validate_crop(crop: np.ndarray) -> np.ndarray:
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("Input crop must be a non-empty 3-channel BGR numpy array.")
    return crop


def _l1_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.flatten().astype(np.float32)
    return vec / (float(np.sum(np.abs(vec))) + 1e-8)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.flatten().astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / (norm + 1e-8)


def _soft_l2_normalize(vec: np.ndarray, tau: float = 0.14) -> np.ndarray:
    """Regularized L2 normalization that damps near-zero achromatic noise while unit-normalizing true chroma."""
    vec = vec.flatten().astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / float(np.sqrt(norm * norm + tau * tau))


def extract_baseline_hsv_features(crop: np.ndarray) -> np.ndarray:
    """
    1. Baseline HSV 2D histogram (18 Hue x 8 Saturation = 144 dims) with green pitch masking.
    Preserves exact compatibility with the baseline Norway-Wales feature extractor.
    """
    crop = _validate_crop(crop)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(
        hsv,
        np.array([35, 40, 30], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    mask: np.ndarray | None = cv2.bitwise_not(grass_mask)
    if cv2.countNonZero(mask) < crop.shape[0] * crop.shape[1] * 0.25:
        mask = None

    histogram = cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256])
    return _l1_normalize(histogram)


def extract_legacy_enhanced_features(crop: np.ndarray) -> np.ndarray:
    """
    Exact reproduction of Cell 14 `enhanced_jersey_features` for fair benchmarking.
    """
    crop = _validate_crop(crop)
    h, w = crop.shape[:2]
    focused = crop[
        int(h * 0.07) : max(int(h * 0.93), 1),
        int(w * 0.10) : max(int(w * 0.90), 1),
    ]
    if focused.size == 0:
        focused = crop
    focused = cv2.resize(focused, (48, 64))

    hsv = cv2.cvtColor(focused, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(focused, cv2.COLOR_BGR2LAB)

    grass = cv2.inRange(
        hsv,
        np.array([35, 40, 30], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    mask: np.ndarray | None = cv2.bitwise_not(grass)
    if cv2.countNonZero(mask) < focused.shape[0] * focused.shape[1] * 0.25:
        mask = None

    hs_hist = _l1_normalize(cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256]))
    ab_hist = _l1_normalize(cv2.calcHist([lab], [1, 2], mask, [12, 12], [0, 256, 0, 256]))
    brightness_hist = _l1_normalize(cv2.calcHist([hsv], [2], mask, [12], [0, 256]))

    features = np.concatenate([
        hs_hist * 0.50,
        ab_hist * 0.35,
        brightness_hist * 0.15,
    ])
    return features.astype(np.float32)


def extract_hsv_lab_features(crop: np.ndarray) -> np.ndarray:
    """
    2. Properly normalized HSV + CIE LAB colour features.
    Fixes three flaws in the legacy enhanced extractor:
      a) Explicitly separates chromatic pixels (S >= 45) from achromatic pixels (white/grey/black, S < 45)
         so low-saturation pixels do not pollute Hue histograms with random noise.
      b) Restricts CIE LAB `a` and `b` histogram ranges to [90, 200] where real kit colours reside,
         preventing all pixels from collapsing into two central bins around 128.
      c) L2-normalizes each histogram block before weighted concatenation so cosine similarity
         respects the intended block weights.
    """
    crop = _validate_crop(crop)
    h, w = crop.shape[:2]
    focused = crop[
        int(h * 0.08) : max(int(h * 0.92), 1),
        int(w * 0.12) : max(int(w * 0.88), 1),
    ]
    if focused.size == 0:
        focused = crop
    focused = cv2.resize(focused, (48, 64), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(focused, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(focused, cv2.COLOR_BGR2LAB)

    fg_mask, _ = compute_grass_mask(focused)
    if cv2.countNonZero(fg_mask) < focused.shape[0] * focused.shape[1] * 0.20:
        fg_mask = np.full(focused.shape[:2], 255, dtype=np.uint8)

    # Separate chromatic mask (S >= 45, V >= 35) vs achromatic mask
    chromatic_cond = cv2.inRange(
        hsv,
        np.array([0, 45, 35], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    )
    chromatic_mask = cv2.bitwise_and(fg_mask, chromatic_cond)
    achromatic_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(chromatic_cond))

    total_fg = float(max(1, cv2.countNonZero(fg_mask)))
    chromatic_ratio = float(cv2.countNonZero(chromatic_mask)) / total_fg
    achromatic_ratio = float(cv2.countNonZero(achromatic_mask)) / total_fg

    # Hue-Saturation on chromatic foreground pixels
    hs_hist = _l2_normalize(
        cv2.calcHist([hsv], [0, 1], chromatic_mask, [18, 6], [0, 180, 45, 256])
    )
    # Value histogram on achromatic pixels (captures white vs grey vs black kits)
    achrom_v_hist = _l2_normalize(
        cv2.calcHist([hsv], [2], achromatic_mask, [10], [0, 256])
    )
    # Full foreground L, a, b 1D histograms over effective jersey gamut
    l_hist = _l2_normalize(cv2.calcHist([lab], [0], fg_mask, [12], [0, 256]))
    a_hist = _l2_normalize(cv2.calcHist([lab], [1], fg_mask, [16], [88, 196]))
    b_hist = _l2_normalize(cv2.calcHist([lab], [2], fg_mask, [16], [88, 196]))

    ratios = np.array([chromatic_ratio, achromatic_ratio], dtype=np.float32)

    combined = np.concatenate([
        hs_hist * 0.35,
        achrom_v_hist * 0.20,
        l_hist * 0.15,
        a_hist * 0.20,
        b_hist * 0.20,
        ratios * 0.25,
    ])
    return _l2_normalize(combined)


def _build_torso_spatial_mask(height: int, width: int) -> np.ndarray:
    """Create an elliptical spatial mask focusing on the central jersey chest area."""
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, int(height * 0.48))
    axes = (max(1, int(width * 0.40)), max(1, int(height * 0.44)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def extract_foreground_masked_features(crop: np.ndarray) -> np.ndarray:
    """
    3. Improved torso foreground masking:
    Combines adaptive pitch-green rejection, elliptical torso-center spatial masking
    (excluding peripheral arms/background), and HSV + LAB colour distributions.
    """
    crop = _validate_crop(crop)
    resized = cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

    grass_fg, _ = compute_grass_mask(resized)
    ellipse_mask = _build_torso_spatial_mask(64, 48)
    combined_mask = cv2.bitwise_and(grass_fg, ellipse_mask)

    # Fallback if combined mask is too sparse
    if cv2.countNonZero(combined_mask) < 48 * 64 * 0.15:
        combined_mask = grass_fg if cv2.countNonZero(grass_fg) >= 48 * 64 * 0.15 else ellipse_mask

    # Base HSV + LAB histograms inside the refined mask
    hs_hist = _l2_normalize(cv2.calcHist([hsv], [0, 1], combined_mask, [18, 6], [0, 180, 0, 256]))
    sv_hist = _l2_normalize(cv2.calcHist([hsv], [1, 2], combined_mask, [6, 8], [0, 256, 0, 256]))
    ab_hist = _l2_normalize(cv2.calcHist([lab], [1, 2], combined_mask, [12, 12], [88, 196, 88, 196]))

    # Extract masked foreground pixels for circular Hue and LAB moments
    fg_bool = combined_mask > 0
    if not np.any(fg_bool):
        fg_bool = np.ones(resized.shape[:2], dtype=bool)

    hsv_pts = hsv[fg_bool].astype(np.float32)
    lab_pts = lab[fg_bool].astype(np.float32)

    # Circular hue encoding: (S/255)*cos(2*pi*H/180), (S/255)*sin(2*pi*H/180)
    angles = (hsv_pts[:, 0] / 180.0) * (2.0 * np.pi)
    sat_norm = hsv_pts[:, 1] / 255.0
    val_norm = hsv_pts[:, 2] / 255.0
    cos_h = sat_norm * np.cos(angles)
    sin_h = sat_norm * np.sin(angles)

    lab_norm = (lab_pts - np.array([128.0, 128.0, 128.0], dtype=np.float32)) / 128.0

    moments = np.array(
        [
            float(np.mean(cos_h)),
            float(np.std(cos_h)),
            float(np.mean(sin_h)),
            float(np.std(sin_h)),
            float(np.mean(sat_norm)),
            float(np.mean(val_norm)),
            float(np.mean(lab_norm[:, 0])),
            float(np.std(lab_norm[:, 0])),
            float(np.mean(lab_norm[:, 1])),
            float(np.std(lab_norm[:, 1])),
            float(np.mean(lab_norm[:, 2])),
            float(np.std(lab_norm[:, 2])),
        ],
        dtype=np.float32,
    )

    feat = np.concatenate([
        hs_hist * 0.30,
        sv_hist * 0.25,
        ab_hist * 0.25,
        _l2_normalize(moments) * 0.35,
    ])
    return _l2_normalize(feat)


def extract_dominant_colour_features(crop: np.ndarray) -> np.ndarray:
    """
    4. Dominant-colour & quantile colour moment descriptor.
    Captures primary and secondary kit colours (e.g. red/white stripes vs solid red vs yellow referee)
    using foreground pixel quantiles and colour-family membership fractions.
    """
    crop = _validate_crop(crop)
    resized = cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

    grass_fg, _ = compute_grass_mask(resized)
    ellipse_mask = _build_torso_spatial_mask(64, 48)
    mask = cv2.bitwise_and(grass_fg, ellipse_mask)
    fg_bool = mask > 0
    if np.count_nonzero(fg_bool) < 25:
        fg_bool = grass_fg > 0
    if np.count_nonzero(fg_bool) < 10:
        fg_bool = np.ones(resized.shape[:2], dtype=bool)

    bgr_pts = resized[fg_bool].astype(np.float32) / 255.0
    hsv_pts = hsv[fg_bool].astype(np.float32)
    lab_pts = (lab[fg_bool].astype(np.float32) - 128.0) / 128.0

    # Quantiles (20th, 50th, 80th percentiles) of normalized BGR and centered LAB
    bgr_q = np.percentile(bgr_pts, [20, 50, 80], axis=0).flatten().astype(np.float32)
    lab_q = np.percentile(lab_pts, [20, 50, 80], axis=0).flatten().astype(np.float32)

    # Colour family proportions on foreground pixels
    h_vals = hsv_pts[:, 0]
    s_vals = hsv_pts[:, 1]
    v_vals = hsv_pts[:, 2]
    n_pts = float(len(h_vals))

    is_dark = v_vals < 65
    is_white = (s_vals < 55) & (v_vals >= 155)
    is_grey = (s_vals < 55) & (v_vals >= 65) & (v_vals < 155)
    is_chromatic = (s_vals >= 55) & (v_vals >= 50)

    is_red = is_chromatic & ((h_vals <= 12) | (h_vals >= 165))
    is_orange_yellow = is_chromatic & (h_vals > 12) & (h_vals <= 36)
    is_green = is_chromatic & (h_vals > 36) & (h_vals <= 92)
    is_cyan_blue = is_chromatic & (h_vals > 92) & (h_vals <= 138)
    is_purple_magenta = is_chromatic & (h_vals > 138) & (h_vals < 165)

    family_fractions = np.array(
        [
            np.sum(is_dark) / n_pts,
            np.sum(is_white) / n_pts,
            np.sum(is_grey) / n_pts,
            np.sum(is_red) / n_pts,
            np.sum(is_orange_yellow) / n_pts,
            np.sum(is_green) / n_pts,
            np.sum(is_cyan_blue) / n_pts,
            np.sum(is_purple_magenta) / n_pts,
        ],
        dtype=np.float32,
    )

    # Split foreground pixels by lightness median to capture two-tone / striped kits
    l_med = float(np.median(lab_pts[:, 0]))
    bright_half = lab_pts[lab_pts[:, 0] >= l_med]
    dark_half = lab_pts[lab_pts[:, 0] < l_med]
    bright_proto = np.mean(bright_half, axis=0) if len(bright_half) > 0 else np.mean(lab_pts, axis=0)
    dark_proto = np.mean(dark_half, axis=0) if len(dark_half) > 0 else np.mean(lab_pts, axis=0)
    two_tone_contrast = np.linalg.norm(bright_proto - dark_proto)

    prototypes = np.concatenate([
        bright_proto.astype(np.float32),
        dark_proto.astype(np.float32),
        np.array([float(two_tone_contrast)], dtype=np.float32),
    ])

    combined = np.concatenate([
        _l2_normalize(family_fractions) * 0.45,
        _l2_normalize(lab_q) * 0.25,
        _l2_normalize(bgr_q) * 0.15,
        _l2_normalize(prototypes) * 0.25,
    ])
    return _l2_normalize(combined)


def _zone_colour_descriptor(
    hsv_zone: np.ndarray,
    lab_zone: np.ndarray,
    mask_zone: np.ndarray,
) -> np.ndarray:
    """Compute a fine-grained HSV + LAB + colour-family descriptor for a torso zone."""
    use_mask: np.ndarray | None = mask_zone
    if cv2.countNonZero(mask_zone) < 8:
        use_mask = None

    h_hist = _l2_normalize(cv2.calcHist([hsv_zone], [0], use_mask, [18], [0, 180]))
    sv_hist = _l2_normalize(cv2.calcHist([hsv_zone], [1, 2], use_mask, [4, 4], [0, 256, 0, 256]))
    ab_hist = _l2_normalize(cv2.calcHist([lab_zone], [1, 2], use_mask, [8, 8], [88, 196, 88, 196]))

    fg = mask_zone > 0
    if np.count_nonzero(fg) < 6:
        fg = np.ones(hsv_zone.shape[:2], dtype=bool)
    lab_pts = (lab_zone[fg].astype(np.float32) - 128.0) / 128.0
    lab_moments = np.concatenate([np.mean(lab_pts, axis=0), np.std(lab_pts, axis=0)]).astype(np.float32)

    return _l2_normalize(
        np.concatenate([
            h_hist * 0.40,
            sv_hist * 0.25,
            ab_hist * 0.25,
            _l2_normalize(lab_moments) * 0.20,
        ])
    )


def extract_spatial_embedding_features(crop: np.ndarray) -> np.ndarray:
    """
    5. Lightweight Spatial Appearance Embedding:
    Combines foreground-masked global HSV+LAB features with a 2-zone vertical torso split
    (upper chest vs lower chest) and Sobel horizontal/vertical stripe energy.
    """
    crop = _validate_crop(crop)
    resized = cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

    grass_fg, _ = compute_grass_mask(resized)
    ellipse_mask = _build_torso_spatial_mask(64, 48)
    mask = cv2.bitwise_and(grass_fg, ellipse_mask)
    if cv2.countNonZero(mask) < 48 * 64 * 0.15:
        mask = grass_fg if cv2.countNonZero(grass_fg) >= 48 * 64 * 0.15 else ellipse_mask

    # Upper chest and lower chest spatial zones
    splits = [(6, 34), (30, 58)]
    zone_vecs = []
    for r0, r1 in splits:
        z_desc = _zone_colour_descriptor(hsv[r0:r1, 6:42], lab[r0:r1, 6:42], mask[r0:r1, 6:42])
        zone_vecs.append(z_desc)

    spatial_vec = _l2_normalize(np.concatenate(zone_vecs))

    # Stripe / texture orientation energy on L channel in center chest
    l_chest = lab[10:54, 8:40, 0].astype(np.float32)
    gx = cv2.Sobel(l_chest, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_chest, cv2.CV_32F, 0, 1, ksize=3)
    ex = float(np.mean(np.abs(gx)))
    ey = float(np.mean(np.abs(gy)))
    stripe_ratio = ex / (ex + ey + 1e-5)
    contrast_std = float(np.std(l_chest)) / 128.0
    texture_vec = _l2_normalize(np.array([stripe_ratio, 1.0 - stripe_ratio, contrast_std], dtype=np.float32))

    fg_masked_vec = extract_foreground_masked_features(crop)

    embedding = np.concatenate([
        fg_masked_vec * 0.65,
        spatial_vec * 0.30,
        texture_vec * 0.10,
    ])
    return _l2_normalize(embedding)


def extract_hybrid_hsv_lab_fg_features(crop: np.ndarray) -> np.ndarray:
    """
    6. Hybrid Chromatic/Achromatic HSV+LAB + Foreground-Masked Torso Descriptor:
    Fuses `hsv_lab` (0.60 weight) with `foreground_masked` (0.40 weight) to preserve
    clean achromatic separation (white/grey/black kits) alongside elliptical central
    torso colour moments.
    """
    hsv_lab_vec = extract_hsv_lab_features(crop)
    fg_vec = extract_foreground_masked_features(crop)
    return _l2_normalize(np.concatenate([hsv_lab_vec * 0.60, fg_vec * 0.40]))


def extract_illumination_invariant_features(
    crop_bgr: np.ndarray,
    h_bins: int = 12,
    s_bins: int = 6,
    v_bins: int = 6,
) -> np.ndarray:
    """
    7. Illumination-Invariant & Glare/Shadow-Robust Torso Appearance Descriptor (Sprint 3.3):
    Designed for harsh stadium floodlights (specular glare + deep lower-torso shadows) and
    daylight colour-temperature variations:
      a) Specular Glare & Printed-Number Trimming: On dark/chromatic kits (`L_p35 < 118`),
         trims high-value low-saturation specular highlights and white shirt numbers (`V > 165, S < 65`)
         occupying `< 38%` of the torso mask; on bright white kits (`L_p75 >= 128, med_S < 32`),
         trims dark printed shirt numbers (`V < 82, S < 45`) occupying `< 48%` of the torso mask.
      b) Deep Shadow Exclusion from Achromatic White Ratio: Separates bright achromatic white
         ratio from light-grey and dark shadow/black achromatic ratios via smooth saturation
         and luminance ramps (eliminating the hard `S = 45` cliff), and gates dark achromatic
         ratio by `V_p75` so shadowed white jerseys with dark numbers never match black kits.
      c) CLAHE / Quantile Lightness & Glare-Trimmed Chromaticity Moments: Combines monotone
         `[q, 1-q]` CIE LAB raw + CLAHE lightness quantiles (`L_p20, L_p40, L_p60, L_p80`),
         top-15%-glare-trimmed normalized RGB chromaticity ratios `(R/(R+G+B), G/(R+G+B), B/(R+G+B))`,
         and soft-regularized CIE LAB `(a*, b*)` medians/IQRs + circular Hue moments.
    """
    crop = _validate_crop(crop_bgr)
    h, w = crop.shape[:2]
    focused = crop[
        int(h * 0.08) : max(int(h * 0.92), 1),
        int(w * 0.12) : max(int(w * 0.88), 1),
    ]
    if focused.size == 0:
        focused = crop
    focused = cv2.resize(focused, (48, 64), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(focused, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(focused, cv2.COLOR_BGR2LAB)

    grass_fg, _ = compute_grass_mask(focused)
    ellipse_mask = _build_torso_spatial_mask(64, 48)
    fg_mask = cv2.bitwise_and(grass_fg, ellipse_mask)
    if cv2.countNonZero(fg_mask) < int(48 * 64 * 0.15):
        fg_mask = grass_fg if cv2.countNonZero(grass_fg) >= int(48 * 64 * 0.15) else ellipse_mask
    if cv2.countNonZero(fg_mask) < 20:
        fg_mask = np.full(focused.shape[:2], 255, dtype=np.uint8)

    fg_bool = fg_mask > 0
    l_vals_raw = lab[:, :, 0][fg_bool].astype(np.float32)
    v_vals_raw = hsv[:, :, 2][fg_bool].astype(np.float32)
    s_vals_raw = hsv[:, :, 1][fg_bool].astype(np.float32)

    # a) Specular Glare & Printed-Number Trimming
    l_p35 = float(np.percentile(l_vals_raw, 35)) if len(l_vals_raw) > 0 else 128.0
    l_p70 = float(np.percentile(l_vals_raw, 70)) if len(l_vals_raw) > 0 else 128.0
    l_p75 = float(np.percentile(l_vals_raw, 75)) if len(l_vals_raw) > 0 else 128.0
    non_bright_s = s_vals_raw[v_vals_raw <= 165] if np.any(v_vals_raw <= 165) else s_vals_raw
    med_s_body = float(np.median(non_bright_s)) if len(non_bright_s) > 0 else 0.0
    med_s_all = float(np.median(s_vals_raw)) if len(s_vals_raw) > 0 else 0.0

    trimmed_mask = fg_mask.copy()
    if len(v_vals_raw) >= 20:
        if l_p35 < 118.0 and (med_s_body >= 25.0 or l_p70 < 120.0):
            glare_cond = (hsv[:, :, 2] > 165) & (hsv[:, :, 1] < 65) & (trimmed_mask > 0)
            glare_frac = float(np.count_nonzero(glare_cond)) / float(len(v_vals_raw))
            if 0.0 < glare_frac < 0.38 and (cv2.countNonZero(trimmed_mask) - np.count_nonzero(glare_cond)) >= 12:
                trimmed_mask[glare_cond] = 0
        elif l_p75 >= 128.0 and med_s_all < 32.0:
            dark_num_cond = (hsv[:, :, 2] < 82) & (hsv[:, :, 1] < 45) & (trimmed_mask > 0)
            dark_frac = float(np.count_nonzero(dark_num_cond)) / float(len(v_vals_raw))
            if 0.0 < dark_frac < 0.48 and (cv2.countNonZero(trimmed_mask) - np.count_nonzero(dark_num_cond)) >= 12:
                trimmed_mask[dark_num_cond] = 0

    trim_bool = trimmed_mask > 0
    if np.count_nonzero(trim_bool) < 10:
        trim_bool = fg_bool
        trimmed_mask = fg_mask

    # CLAHE on L* channel to normalize local shadow gradients
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_clahe = clahe.apply(lab[:, :, 0])

    v_trim = hsv[:, :, 2][trim_bool].astype(np.float32)
    s_trim = hsv[:, :, 1][trim_bool].astype(np.float32)
    h_trim = hsv[:, :, 0][trim_bool].astype(np.float32)
    l_trim = lab[:, :, 0][trim_bool].astype(np.float32)
    a_trim = lab[:, :, 1][trim_bool].astype(np.float32)
    b_trim = lab[:, :, 2][trim_bool].astype(np.float32)
    l_clahe_trim = l_clahe[trim_bool].astype(np.float32)
    n_trim = float(max(1, len(v_trim)))

    # b) Illuminant-Invariant Chromatic / Achromatic Partition & Deep Shadow Separation
    lab_chroma_full = np.hypot(
        lab[:, :, 1].astype(np.float32) - 128.0,
        lab[:, :, 2].astype(np.float32) - 128.0,
    )
    h_full = hsv[:, :, 0]
    s_full = hsv[:, :, 1]
    v_full = hsv[:, :, 2]

    is_mild_warm_daylight_cast = (
        (h_full >= 6)
        & (h_full <= 38)
        & (s_full < 62)
        & (v_full >= 85)
        & (lab_chroma_full < 12.5)
    )
    chrom_bool = (
        trim_bool
        & (v_full >= 35)
        & (
            (s_full >= 58)
            | ((s_full >= 26) & (~is_mild_warm_daylight_cast) & ((v_full < 122) | (lab[:, :, 2] <= 126)))
        )
        & (~is_mild_warm_daylight_cast)
    )
    chromatic_mask = np.where(chrom_bool, np.uint8(255), np.uint8(0))
    achromatic_mask = np.where(trim_bool & (~chrom_bool), np.uint8(255), np.uint8(0))

    s_eff = np.where(
        (h_trim >= 6)
        & (h_trim <= 38)
        & (s_trim < 62)
        & (v_trim >= 85)
        & (np.hypot(a_trim - 128.0, b_trim - 128.0) < 12.5),
        s_trim * 0.45,
        s_trim,
    )

    w_achrom = np.clip((62.0 - s_eff) / 36.0, 0.0, 1.0)
    w_chrom = 1.0 - w_achrom
    strong_chrom_w = np.clip((s_eff - 40.0) / 45.0, 0.0, 1.0) * np.clip((v_trim - 30.0) / 25.0, 0.0, 1.0)

    v_p75_trim = float(np.percentile(v_trim, 75)) if len(v_trim) > 0 else 128.0
    dark_kit_gate = float(np.clip((125.0 - v_p75_trim) / 40.0, 0.0, 1.0))

    bright_achrom_ratio = float(np.mean(w_achrom * np.clip((v_trim - 110.0) / 45.0, 0.0, 1.0)))
    light_achrom_ratio = float(np.mean(w_achrom * np.clip((v_trim - 75.0) / 40.0, 0.0, 1.0)))
    dark_achrom_ratio = float(np.mean(w_achrom * np.clip((78.0 - v_trim) / 35.0, 0.0, 1.0))) * dark_kit_gate
    strong_chrom_ratio = float(np.mean(strong_chrom_w))
    soft_chrom_ratio = float(np.mean(w_chrom))

    chrom_occ = float(np.sqrt(np.clip(float(cv2.countNonZero(chromatic_mask)) / n_trim, 0.0, 1.0)))
    achrom_occ = float(np.sqrt(np.clip(float(cv2.countNonZero(achromatic_mask)) / n_trim, 0.0, 1.0)))

    hs_hist = (
        _l2_normalize(cv2.calcHist([hsv], [0, 1], chromatic_mask, [int(h_bins), int(s_bins)], [0, 180, 26, 256]))
        * chrom_occ
    )
    achrom_v_hist = (
        _l2_normalize(cv2.calcHist([hsv], [2], achromatic_mask, [int(v_bins)], [0, 256])) * achrom_occ
    )

    l_hist = _l2_normalize(cv2.calcHist([lab], [0], trimmed_mask, [12], [0, 256]))
    a_hist = _l2_normalize(cv2.calcHist([lab], [1], trimmed_mask, [16], [88, 196]))
    b_hist = _l2_normalize(cv2.calcHist([lab], [2], trimmed_mask, [16], [88, 196]))
    # Joint 2D (L*, a*) and (L*, b*) histograms prevent dark low-saturation kits
    # (e.g. floodlit dark-navy/black in shadow) from sharing neutral (a*=128, b*=128)
    # histogram bins with bright white kits, while 1D a_hist/b_hist preserve strong
    # chromatic separation for maroon/red/yellow goalkeepers and referees!
    la_hist = _l2_normalize(cv2.calcHist([lab], [0, 1], trimmed_mask, [6, 8], [0, 256, 88, 196]))
    lb_hist = _l2_normalize(cv2.calcHist([lab], [0, 2], trimmed_mask, [6, 8], [0, 256, 88, 196]))

    # c) Quantile Lightness & Glare-Trimmed Normalized Chromaticity + CIE LAB Moments
    lq_raw = np.percentile(l_trim, [20, 40, 60, 80]).astype(np.float32) / 255.0
    lq_clahe = np.percentile(l_clahe_trim, [25, 50, 75]).astype(np.float32) / 255.0
    l_quantiles = _l2_normalize(np.concatenate([lq_raw, 1.0 - lq_raw, lq_clahe, 1.0 - lq_clahe]))

    bgr_trim = focused[trim_bool].astype(np.float32)
    v_cut = float(np.percentile(v_trim, 85)) if len(v_trim) >= 8 else 255.0
    non_glare = (v_trim <= max(v_cut, 60.0)) & (v_trim >= 20.0)
    bgr_ng = bgr_trim[non_glare] if np.count_nonzero(non_glare) >= 5 else bgr_trim
    rgb_sum = np.sum(bgr_ng, axis=1, keepdims=True) + 1e-5
    b_rat = bgr_ng[:, 0] / rgb_sum[:, 0]
    g_rat = bgr_ng[:, 1] / rgb_sum[:, 0]
    r_rat = bgr_ng[:, 2] / rgb_sum[:, 0]

    med_lab_chroma = float(np.median(np.hypot(a_trim - 128.0, b_trim - 128.0)))
    med_h = float(np.median(h_trim))
    warm_cast_damp = (
        0.25
        if (
            6.0 <= med_h <= 38.0
            and med_lab_chroma < 11.5
            and float(np.median(v_trim)) >= 92.0
            and float(np.median(s_trim)) < 58.0
        )
        else 1.0
    )

    a_med = ((float(np.median(a_trim)) - 128.0) / 48.0) * warm_cast_damp
    b_med = ((float(np.median(b_trim)) - 128.0) / 48.0) * warm_cast_damp
    l_med_centered = (float(np.median(l_trim)) - 128.0) / 64.0
    a_iqr = float(np.percentile(a_trim, 75) - np.percentile(a_trim, 25)) / 64.0
    b_iqr = float(np.percentile(b_trim, 75) - np.percentile(b_trim, 25)) / 64.0

    angles = (h_trim / 180.0) * (2.0 * np.pi)
    s_norm = (s_eff / 255.0) * warm_cast_damp
    cos_h = float(np.median(s_norm * np.cos(angles)))
    sin_h = float(np.median(s_norm * np.sin(angles)))

    chroma_and_lab_moments = _soft_l2_normalize(
        np.array(
            [
                float(np.median(r_rat) - 1.0 / 3.0) * 3.0 * warm_cast_damp,
                float(np.median(g_rat) - 1.0 / 3.0) * 3.0 * warm_cast_damp,
                float(np.median(b_rat) - 1.0 / 3.0) * 3.0 * warm_cast_damp,
                float(np.median(b_rat - r_rat)) * 3.0 * warm_cast_damp,
                l_med_centered * 0.55,
                a_med * 1.25,
                b_med * 1.25,
                a_iqr * 0.4,
                b_iqr * 0.4,
                cos_h * 1.5,
                sin_h * 1.5,
            ],
            dtype=np.float32,
        ),
        tau=0.14,
    )

    # Separate bright vs dark achromatic lightness balance explicitly
    ratios = np.array(
        [
            strong_chrom_ratio,
            soft_chrom_ratio,
            bright_achrom_ratio * 1.20,
            light_achrom_ratio * 1.00,
            dark_achrom_ratio * 1.20,
        ],
        dtype=np.float32,
    )

    combined = np.concatenate(
        [
            hs_hist * 0.30,
            achrom_v_hist * 0.20,
            l_hist * 0.18,
            a_hist * 0.18,
            b_hist * 0.18,
            la_hist * 0.20,
            lb_hist * 0.20,
            l_quantiles * 0.18,
            chroma_and_lab_moments * 0.24,
            ratios * 0.24,
        ]
    )
    return _l2_normalize(combined)


FEATURE_EXTRACTORS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "baseline_hsv": extract_baseline_hsv_features,
    "legacy_enhanced": extract_legacy_enhanced_features,
    "hsv_lab": extract_hsv_lab_features,
    "foreground_masked": extract_foreground_masked_features,
    "dominant_colour": extract_dominant_colour_features,
    "spatial_embedding": extract_spatial_embedding_features,
    "hybrid_hsv_lab_fg": extract_hybrid_hsv_lab_fg_features,
    "illumination_invariant": extract_illumination_invariant_features,
}


def get_feature_extractor(method_name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Retrieve a registered feature extraction function by name."""
    if method_name not in FEATURE_EXTRACTORS:
        raise ValueError(
            f"Unknown feature method '{method_name}'. Available methods: {list(FEATURE_EXTRACTORS.keys())}"
        )
    return FEATURE_EXTRACTORS[method_name]

