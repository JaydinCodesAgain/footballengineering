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


FEATURE_EXTRACTORS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "baseline_hsv": extract_baseline_hsv_features,
    "legacy_enhanced": extract_legacy_enhanced_features,
    "hsv_lab": extract_hsv_lab_features,
    "foreground_masked": extract_foreground_masked_features,
    "dominant_colour": extract_dominant_colour_features,
    "spatial_embedding": extract_spatial_embedding_features,
}


def get_feature_extractor(method_name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Retrieve a registered feature extraction function by name."""
    if method_name not in FEATURE_EXTRACTORS:
        raise ValueError(
            f"Unknown feature method '{method_name}'. Available methods: {list(FEATURE_EXTRACTORS.keys())}"
        )
    return FEATURE_EXTRACTORS[method_name]

