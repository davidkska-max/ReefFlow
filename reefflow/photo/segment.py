"""Rock segmentation per photo, and mask editing operations (pure functions).

Masks are boolean H x W arrays (True = rock), indexed ``mask[v, u]``. A second,
optional boolean ``ignore`` mask per view marks pixels that carry no
information (equipment in front of the rock, frame trim, glare); carving
treats them as "unknown" and never removes voxels because of them.

Automatic suggestion (DECISIONS D-603)
--------------------------------------
GrabCut (OpenCV, classical, always available) initialised from the
calibration plus a background colour model:

1. every pixel is back-projected through the interface; pixels whose ray never
   enters the water column (frame, room, sand face behind the pane, air above
   the water) are definite background;
2. background colours are learned (k-means in CIE-Lab) from pixels whose ray
   cannot meet rock: rays that miss the "rock-able" box (inset ``wall_margin``
   from the glass and ``top_margin`` below the surface) and, in the top view,
   rays ending on the sand next to the glass;
3. tank pixels in the rock-able region whose colour is far from every
   background cluster (CIE-Lab Delta-E > ``colour_threshold``, default 14) are
   probable rock; the rest of the tank is probable background;
4. GrabCut refines this (colour GMMs + contrast-sensitive smoothness);
5. for a photo set (:func:`auto_segment_views`) the top view is done first and
   its prism, reprojected into each side view, replaces the geometric margins:
   background colours are then learned from every tank pixel outside the
   prism (back wall, side walls, open sand), and only pixels inside it can be
   probable rock. If GrabCut still swallows the whole candidate region the
   threshold is raised 1.4x and it retries (up to 3 times).

The default threshold (14) was chosen on synthetic renders (seeds 1-3); the
validation uses a held-out render seed. Real photos (shadows, coralline-covered
glass, equipment) will often need a few user strokes.

A loose purely geometric prior (all rock-able pixels 'probable rock') was
tried first and fails: the background model then has no samples of the back
wall / open sand and GrabCut labels the whole tank as rock. If a SAM-family model is installed (models.SamSegmenter) and requested,
it is used instead; otherwise the result says "SAM not installed; used GrabCut".
"""
from __future__ import annotations

import numpy as np

from .camera import View

GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD = 0, 1, 2, 3


# ----------------------------------------------------------------------
# geometric prior from the calibration
# ----------------------------------------------------------------------
def _ray_box(o, d, lo, hi):
    """Slab test: does o + t d (t >= 0) hit the box? (N,) bool."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(d) < 1e-15, 1e-15, d)
        t1 = (np.asarray(lo) - o) * inv
        t2 = (np.asarray(hi) - o) * inv
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    return tmax >= np.maximum(tmin, 0.0)


def view_region_masks(view: View, wall_margin: float = 0.03, top_margin: float = 0.06,
                      step: int = 2, floor_band: float = 0.03) -> dict:
    """Per-pixel geometric masks from the calibrated view.

    Returns bool images:
      'tank'      rays that enter the water column;
      'rockable'  rays that pass through the rock-able box (inset ``wall_margin``
                  from the glass, ``top_margin`` below the surface);
      'bg_sample' pixels used to learn the background colours: tank & ~rockable,
                  plus (top view only) rays ending on the sand within
                  ``wall_margin`` of the glass -- seen from above these rays cross
                  the rock-able box, so without them the background model would
                  contain walls but no sand.
      'floor'     rays that come within ``floor_band`` of the sand inside the tank;
      'sand_hit'  side views: rays ending on the sand inside the tank;
      'sand_near' side views: those ending within ``wall_margin`` of the near pane.
    Computed on every ``step``-th pixel and upsampled (nearest)."""
    import cv2
    cam, T = view.camera, view.tank
    W, H = cam.width, cam.height
    us = np.arange(0, W, step, dtype=float)
    vs = np.arange(0, H, step, dtype=float)
    U_, V_ = np.meshgrid(us, vs)
    uv = np.column_stack([U_.ravel(), V_.ravel()])
    r = view.back_project(uv)
    water = r["valid"] & (r["n_inner"] == T.n_water) if view.refraction else r["valid"]
    E, d = r["entry"], r["dir"]
    if view.kind != "top":            # entry must be in the water column, not the sand face
        water &= (E[:, 2] >= T.sand_depth) & (E[:, 2] <= T.water_surface)
    lo = np.array([wall_margin, wall_margin, T.sand_depth])
    hi = np.array([T.length - wall_margin, T.width - wall_margin, T.water_surface - top_margin])
    rockable = water & _ray_box(E, d, lo, hi)
    bg = water & ~rockable
    # 'floor': rays that, inside the tank, come within `floor_band` of the sand (rock
    # rests on the sand, so every rock blob in a side view reaches these pixels)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(d) < 1e-15, 1e-15, d)
        t1 = (np.array([0.0, 0.0, T.sand_depth]) - E) * inv
        t2 = (np.array([T.length, T.width, T.water_surface]) - E) * inv
    t_exit = np.maximum(np.min(np.maximum(t1, t2), axis=1), 0.0)
    z_low = np.minimum(E[:, 2], E[:, 2] + t_exit * d[:, 2])
    floor = water & (z_low < T.sand_depth + floor_band)
    # 'sand_hit': side-view rays that end on the sand inside the tank (the sand is rough
    # too, so there the texture cue also needs a colour difference); 'sand_near': those
    # ending within wall_margin of the near pane (sand colour samples)
    sand_hit = np.zeros_like(water)
    sand_near = np.zeros_like(water)
    if view.kind != "top":
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = (T.sand_depth - E[:, 2]) / np.where(d[:, 2] < -1e-12, d[:, 2], -1e-12)
        Xs = E + np.maximum(ts, 0)[:, None] * d
        on = ((Xs[:, 0] >= 0) & (Xs[:, 0] <= T.length) & (Xs[:, 1] >= 0) & (Xs[:, 1] <= T.width)
              & (d[:, 2] < -1e-12) & (ts <= t_exit + 1e-9))
        sand_hit = water & on
        sand_near = sand_hit & ((Xs @ view.normal - view.planes[-1]) < wall_margin)
    if view.kind == "top":
        with np.errstate(divide="ignore", invalid="ignore"):
            t_sand = (T.sand_depth - E[:, 2]) / np.where(d[:, 2] < -1e-12, d[:, 2], -1e-12)
        X = E + t_sand[:, None] * d
        on_floor = (X[:, 0] >= 0) & (X[:, 0] <= T.length) & (X[:, 1] >= 0) & (X[:, 1] <= T.width)
        near = (np.minimum(X[:, 0], T.length - X[:, 0]) < wall_margin) |                (np.minimum(X[:, 1], T.width - X[:, 1]) < wall_margin)
        bg |= water & on_floor & near
    out = {}
    for k, m in (("tank", water), ("rockable", rockable), ("bg_sample", bg), ("floor", floor),
                 ("sand_hit", sand_hit), ("sand_near", sand_near)):
        small = m.reshape(len(vs), len(us)).astype(np.uint8)
        out[k] = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) if step > 1             else small.astype(bool)
    return out


def background_distance(image, sample_mask, region_mask, k: int = 8, seed: int = 0) -> np.ndarray:
    """Colour distance (CIE-Lab Delta-E 1976) of every pixel in ``region_mask`` to the
    nearest of ``k`` k-means background colours learned from ``sample_mask``.
    The image is blurred (sigma 1.5 px) first to suppress sensor noise. 0 outside region."""
    import cv2
    lab = cv2.cvtColor(cv2.GaussianBlur(np.asarray(image, np.uint8)[..., :3], (0, 0), 1.5),
                       cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0                      # OpenCV 8-bit L is scaled to 0..255
    S = lab[sample_mask]
    if len(S) < 10 * k:
        raise ValueError("too few background sample pixels for the colour model")
    S = S[:: max(1, len(S) // 60000)].copy()
    cv2.setRNGSeed(int(seed))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, cent = cv2.kmeans(S, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    px = lab[region_mask]
    dist = np.min([np.linalg.norm(px - c, axis=1) for c in cent], axis=0)
    out = np.zeros(region_mask.shape, np.float32)
    out[region_mask] = dist
    return out


def auto_init_labels(view: View, image=None, guide=None, wall_margin: float = 0.03,
                     top_margin: float = 0.06, colour_threshold: float = 14.0) -> np.ndarray:
    """GrabCut label image (uint8, GC_* codes); see module doc.

    guide: optional bool image of pixels where rock *can* be, e.g. the top-view
    prism reprojected into a side view (:func:`guide_from_views`). With a guide,
    background colours are learned from all tank pixels outside it.
    colour_threshold: Delta-E (Lab) above which a pixel is 'probable rock'.
    Without an image the whole rock-able region is GC_PR_FGD (weak prior)."""
    reg = view_region_masks(view, wall_margin, top_margin)
    lab = np.full(reg["tank"].shape, GC_BGD, np.uint8)
    lab[reg["tank"]] = GC_PR_BGD
    cand, sample = reg["rockable"], reg["bg_sample"]
    if guide is not None:
        g = np.asarray(guide, bool)
        cand = reg["tank"] & g
        sample = reg["tank"] & ~g
    if image is None:
        lab[cand] = GC_PR_FGD
        return lab
    D = background_distance(image, sample, reg["tank"])
    fg = cand & (D > colour_threshold)
    if not fg.any():
        raise ValueError("no pixel differs from the background colours: draw a rock stroke or a box")
    lab[fg] = GC_PR_FGD
    return lab


# ----------------------------------------------------------------------
# texture cue for side views (DECISIONS D-839)
# ----------------------------------------------------------------------
def _px(image_shape, at2000: float) -> float:
    """A length given in px for a 2000-px-wide photo, scaled to this photo."""
    return at2000 * max(image_shape[:2]) / 2000.0


def sensitivity_params(sensitivity: float = 50.0) -> dict:
    """What the 0-100 'Sensitivity' setting changes (D-840); 50 = the default suggestion.

    Higher finds more rock, lower less. It works like "expand / contract" with an edge
    test, in a band around the default outline, so it cannot flood open water (a
    global threshold change did: x0.7 filled the tank on a real photo):
      s > 50: pixels within grow_px (up to 40 px per 2000 px at s = 100) of the
              outline are added if they look like rock: roughness > Otsu x 1.25**k,
              or a colour difference from the water/sand samples > dE 14 x 1.5**k,
              k = (50 - s) / 50; only parts connected to the outline count
      s < 50: pixels within shrink_px (up to 15 px per 2000 px at s = 0) of the edge
              are dropped unless rougher than Otsu x 1.25**k (up to x1.25)
    The base suggestion is the same for every setting, so the result changes
    monotonically and can be recomputed from a cached base in about a second.
    """
    sv = float(np.clip(sensitivity, 0.0, 100.0))
    k = (50.0 - sv) / 50.0
    return {"k": k, "edge_factor": 1.25 ** k,
            "colour_threshold": 14.0 * 1.5 ** k,
            "grow_px": max(0.0, sv - 50.0) * 0.8, "shrink_px": max(0.0, 50.0 - sv) * 0.3}


def texture_energy(image, scale: float | None = None, speck_px: float = 5.0) -> np.ndarray:
    """Local roughness: Gaussian-averaged gradient magnitude divided by the local
    brightness (+8 grey levels), so dim rock in shadow still scores. Porous rock is
    rough; open water, the back glass and smooth equipment are not. ``scale``
    defaults to 6 px per 2000 px of photo.

    A median filter (``speck_px`` per 2000 px, 0 = off) runs first: micro-bubbles
    and floating particles are bright specks a few px across that made dark open
    water look as rough as rock (D-842); rock texture is coarser and survives it.
    Only used when the kernel is at least 5 px (photos of ~1800 px and up)."""
    import cv2
    s = scale or max(1.0, _px(image.shape, 6.0))
    g8 = cv2.cvtColor(np.asarray(image, np.uint8)[..., :3], cv2.COLOR_RGB2GRAY)
    if speck_px:
        kk = int(round(_px(image.shape, speck_px))) | 1
        if kk >= 5:     # photos >= ~1800 px (the app stores 2000 px); smaller images
            g8 = cv2.medianBlur(g8, kk)     # blur specks anyway, and a 3-px median ate fine rock texture
    g = cv2.GaussianBlur(g8.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    e = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), s)
    return e / (cv2.GaussianBlur(g, (0, 0), s) + 8.0)


def seam_band(view: View, width_px: float) -> np.ndarray:
    """Bool image of the projected vertical tank corners (inside seams and outside
    edges), ``width_px`` wide: glass edges glow under reef lights and look rough."""
    import cv2
    T, cam = view.tank, view.camera
    out = np.zeros((cam.height, cam.width), np.uint8)
    g = T.glass
    t = np.linspace(0, 1, 33)[:, None]
    for x, y in [(0, 0), (T.length, 0), (T.length, T.width), (0, T.width),
                 (-g, -g), (T.length + g, -g), (T.length + g, T.width + g), (-g, T.width + g)]:
        a, b = np.array([x, y, 0.0]), np.array([x, y, T.height])
        uv, ok = view.project(a + t * (b - a))
        ok &= cam.in_image(uv, margin=-cam.width)
        run = np.round(uv[ok]).astype(np.int32)
        if len(run) > 1:
            cv2.polylines(out, [run], False, 1, thickness=max(1, int(round(width_px))))
    return out.astype(bool)


def sand_lightness_ratio(image, sand_hit) -> np.ndarray:
    """Float image: lightness / local sand lightness on sand-ending rays (1 elsewhere).
    Below 0.85 counts as 'clearly darker than the sand' (:func:`darker_than_sand`)."""
    import cv2
    sh = np.asarray(sand_hit, bool)
    out = np.ones(sh.shape, np.float32)
    if np.count_nonzero(sh) < 100:
        return out
    L = cv2.GaussianBlur(cv2.cvtColor(np.asarray(image, np.uint8)[..., :3], cv2.COLOR_RGB2LAB)[..., 0]
                         .astype(np.float32), (0, 0), max(1.0, _px(sh.shape, 3.0)))
    H, W = sh.shape
    bw = max(4, int(round(_px(sh.shape, 50.0))))
    nb = (W + bw - 1) // bw
    vv, uu = np.nonzero(sh)
    vals = L[vv, uu]
    b = uu // bw
    half = 3                                   # +-3 bins of 50 px = +-150 px
    ref_b = np.full(nb, np.nan)
    order = np.argsort(b, kind="stable")
    bs, vs = b[order], vals[order]
    starts = np.searchsorted(bs, np.arange(nb + 1))
    for i in range(nb):
        lo, hi = starts[max(0, i - half)], starts[min(nb, i + half + 1)]
        if hi - lo > 50:
            ref_b[i] = np.percentile(vs[lo:hi], 80)
    if np.all(np.isnan(ref_b)):
        return out
    ok = ~np.isnan(ref_b)
    ref_b = np.interp(np.arange(nb), np.nonzero(ok)[0], ref_b[ok])
    ref = np.repeat(ref_b, bw)[:W]
    out[sh] = (L / np.maximum(ref[None, :], 1.0))[sh]
    return out


def darker_than_sand(image, sand_hit, factor: float = 0.85) -> np.ndarray:
    """Bool image: pixels on sand-ending rays clearly darker than the sand around them.

    Rubble lying on the sand is about as rough as the sand and, under blue light,
    about the same colour, but darker (on a real photo L* 108-118 against 138-152).
    The reference is the 80th percentile of the blurred Lab lightness over the
    sand-ending pixels in a +-150 px (per 2000 px) wide column window; a pixel is
    'darker' below ``factor`` x that (D-842)."""
    return np.asarray(sand_hit, bool) & (sand_lightness_ratio(image, sand_hit) < factor)


def keep_touching(mask, ref) -> np.ndarray:
    """Connected components of ``mask`` (8-connected) that overlap ``ref``."""
    import cv2
    m = np.asarray(mask, bool)
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    hit = np.unique(lab[np.asarray(ref, bool) & m])
    return np.isin(lab, hit[hit > 0])


def texture_init(view: View, image, guide=None, strict: float = 1.0, wall_margin: float = 0.03,
                 top_margin: float = 0.06, colour_threshold: float | None = None,
                 sensitivity: float = 50.0) -> dict:
    """GrabCut initialisation for a side view from roughness (D-839).

    1. texture energy (:func:`texture_energy`), zeroed on the projected vertical tank
       corners (:func:`seam_band`, 24 px per 2000 px);
    2. threshold: Otsu on the energy of the tank pixels, times ``strict``;
    3. seeds: rough pixels in the candidate region (rock-able box, or the guide),
       opened with a 15-px disk (per 2000 px) so thin edges, pipes and cables drop
       out, keeping only blobs that reach the sand ('floor' region): rock rests on the
       sand, equipment hangs from the rim. Where the ray ends on the sand (which is
       rough too) a seed pixel must also differ in colour from the background and sand
       samples by more than ``colour_threshold`` (Delta-E);
    4. labels: outside the candidate region definite background; candidate region
       probable background, seeds probable rock.
    Returns {labels, features, floor, threshold, seed_fraction}. ``features`` is the
    3-channel image GrabCut runs on: (roughness at a 2.5-px scale, grey level, colour
    distance to the learned background), because under blue reef light colour alone
    barely separates rock from water; the finer roughness keeps the outline tight."""
    import cv2
    # the sensitivity does not change this base suggestion (GrabCut reacts unpredictably
    # to small seed changes); it only moves the outline afterwards (adjust_outline)
    sp = sensitivity_params(sensitivity)
    if colour_threshold is None:
        colour_threshold = 14.0
    reg = view_region_masks(view, wall_margin, top_margin)
    tank = reg["tank"]
    cand = reg["rockable"] if guide is None else (tank & np.asarray(guide, bool))
    tex = texture_energy(image)
    tex[seam_band(view, _px(image.shape, 24.0))] = 0.0
    tv = tex[tank]
    if tv.size < 100:
        raise ValueError("the calibrated tank region is too small in this photo")
    top = float(np.percentile(tv, 99.5)) or 1.0
    q = np.clip(tv / top * 255, 0, 255).astype(np.uint8)
    thr = cv2.threshold(q.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0 * top * strict
    r = max(1, int(round(_px(image.shape, 7.5))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    sample = (tank & ~cand) | (cand & (tex < 0.5 * thr)) | reg["sand_near"]
    D = (background_distance(image, sample, tank) if np.count_nonzero(sample) >= 100
         else np.zeros(tank.shape, np.float32))
    # where the ray ends on the sand (rough too), rock must also differ in colour or be
    # clearly darker than the sand around it (rubble; D-842)
    sh = reg["sand_hit"]
    ratio = sand_lightness_ratio(image, sh)
    dark = sh & (ratio < 0.85)
    sand_rock = sh & (tex > 0.8 * thr) & ((D > colour_threshold) | dark)
    rough = cand & (((tex > thr) & ~sh) | sand_rock)
    seed = cv2.morphologyEx(rough.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    floor = reg["floor"] & tank
    seed = keep_touching(seed, floor)
    # GrabCut may only move the outline within 20 px (per 2000 px) of the seeds, or
    # inside areas they enclose once gaps up to 60 px are closed (smooth rock faces
    # ringed by rough edges, arches): with
    # the whole tank as 'probable background' its colour model pulled in dark water
    # behind the rock and the sand (D-842)
    rb = max(1, int(round(_px(image.shape, 20.0))))
    kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rb + 1, 2 * rb + 1))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6 * rb + 1, 6 * rb + 1))
    hull = fill_holes(cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, kc).astype(bool))
    band = cv2.dilate(hull.astype(np.uint8), kb).astype(bool) & cand
    lab = np.full(tank.shape, GC_BGD, np.uint8)
    lab[band] = GC_PR_BGD
    lab[seed] = GC_PR_FGD
    grey = cv2.cvtColor(np.asarray(image, np.uint8)[..., :3], cv2.COLOR_RGB2GRAY)
    fine = texture_energy(image, max(1.0, _px(image.shape, 2.5)))
    fine[seam_band(view, _px(image.shape, 24.0))] = 0.0
    ftop = float(np.percentile(fine[tank], 99.5)) or 1.0
    feat = np.dstack([np.clip(fine / ftop * 255, 0, 255), grey, np.clip(D * 4, 0, 255)]).astype(np.uint8)
    edge = {"tex": tex, "thr": float(thr / max(strict, 1e-9)), "D": D, "sand_hit": sh, "sand_ratio": ratio,
            "cand": cand, "floor": floor, "params": sp}
    return {"labels": lab, "features": feat, "floor": floor, "threshold": float(thr), "edge": edge,
            "seed_fraction": float(np.count_nonzero(seed) / max(np.count_nonzero(cand), 1))}


def guide_from_views(view: View, views, masks, dx: float = 0.01, dilate_px: int = 15) -> np.ndarray:
    """Where can rock appear in ``view``, given masks already accepted in other views?

    Carves a coarse visual hull from (views, masks) -- typically just the top view,
    which gives a prism -- reprojects it into ``view`` and dilates it."""
    import cv2
    from .carve import VoxelGrid, carve_visual_hull
    from .compare import reproject_silhouette
    grid = VoxelGrid.for_tank(view.tank, dx)
    res = carve_visual_hull(views, masks, grid, tolerance_px=2)
    sil = reproject_silhouette(view, res.occupancy, grid).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    return cv2.dilate(sil, k).astype(bool)


# ----------------------------------------------------------------------
# GrabCut / SAM
# ----------------------------------------------------------------------
def _stroke_labels(lab, strokes, value, radius):
    import cv2
    for s in strokes or []:
        pts = np.round(np.atleast_2d(np.asarray(s, float))).astype(np.int32)
        if len(pts) == 1:
            cv2.circle(lab, tuple(int(x) for x in pts[0]), int(radius), int(value), -1)
        else:
            cv2.polylines(lab, [pts], False, int(value), thickness=max(1, int(2 * radius)))
    return lab


def grabcut_segment(image, init_labels=None, rect=None, fg_strokes=None, bg_strokes=None,
                    iterations: int = 5, stroke_radius: int = 6, seed: int = 0) -> np.ndarray:
    """GrabCut on an RGB uint8 image. Give ``init_labels`` (GC_* codes) or ``rect``
    [u0, v0, u1, v1]; strokes are lists of [[u, v], ...] polylines forced to
    foreground / background. Returns a bool mask."""
    import cv2
    img = np.ascontiguousarray(np.asarray(image)[..., :3].astype(np.uint8))
    H, W = img.shape[:2]
    if init_labels is not None:
        lab = np.asarray(init_labels, np.uint8).copy()
    elif rect is not None:
        u0, v0, u1, v1 = (int(round(x)) for x in rect)
        lab = np.full((H, W), GC_BGD, np.uint8)
        lab[max(v0, 0):min(v1, H), max(u0, 0):min(u1, W)] = GC_PR_FGD
    else:
        raise ValueError("give init_labels or rect")
    lab = _stroke_labels(lab, fg_strokes, GC_FGD, stroke_radius)
    lab = _stroke_labels(lab, bg_strokes, GC_BGD, stroke_radius)
    if not np.any((lab == GC_FGD) | (lab == GC_PR_FGD)) or not np.any((lab == GC_BGD) | (lab == GC_PR_BGD)):
        raise ValueError("GrabCut needs both foreground and background pixels in the initialisation")
    cv2.setRNGSeed(int(seed))
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, lab, None, bgd, fgd, int(iterations), cv2.GC_INIT_WITH_MASK)
    return (lab == GC_FGD) | (lab == GC_PR_FGD)


def adjust_outline(mask, edge: dict) -> np.ndarray:
    """Grow or shrink a texture-cue mask near its outline by the sensitivity (D-840)."""
    import cv2
    sp, m = edge["params"], np.asarray(mask, bool)
    if not m.any() or (sp["grow_px"] <= 0 and sp["shrink_px"] <= 0):
        return m
    tex, thr = edge["tex"], edge["thr"]
    if sp["grow_px"] > 0:
        r = max(1, int(round(_px(m.shape, sp["grow_px"]))))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        band = cv2.dilate(m.astype(np.uint8), k).astype(bool) & ~m & edge["cand"]
        # openings (arches, gaps) bigger than the hole-fill size stay open: growing from
        # all sides closed them at s >= 75 on a real photo (D-842)
        big_holes = fill_holes(m) & ~m & ~fill_holes(m, max_area=int(1e-3 * m.size))
        band &= ~big_holes
        rocklike = (tex > thr * sp["edge_factor"]) | (edge["D"] > sp["colour_threshold"])
        sh = edge.get("sand_hit")
        if sh is not None:      # on the sand roughness alone is not enough (D-842)
            # sensitivity loosens the darkness test (0.80 at s = 0 ... 0.90 at s = 100) and the
            # roughness; the colour test on the sand stays at dE 14 (lower lets sand in)
            sd = edge.get("sand_ratio", np.ones(sh.shape, np.float32)) < 0.85 * 1.06 ** (-sp["k"])
            on_sand = (tex > 0.8 * thr * sp["edge_factor"]) & (sd | (edge["D"] > max(14.0, sp["colour_threshold"])))
            rocklike = np.where(sh, on_sand, rocklike)
        grown = m | (band & rocklike)
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grown = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_OPEN, ko).astype(bool) | m
        return keep_touching(grown, m)
    r = max(1, int(round(_px(m.shape, sp["shrink_px"]))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    rim = m & ~cv2.erode(m.astype(np.uint8), k).astype(bool)
    out = m & ~(rim & (tex < thr * sp["edge_factor"]))
    return keep_touching(out, edge["floor"])


def finish_texture_mask(base, edge: dict, sensitivity: float = 50.0, cleanup_min_area: int = 200) -> np.ndarray:
    """Base texture-cue mask -> final mask: sensitivity adjustment, then a 4-px opening
    (per 2000 px) to drop thin strips along glass/sand edges, specks under 0.067 % of
    the photo removed and holes under 0.1 % filled. Cheap (no GrabCut)."""
    import cv2
    e = dict(edge)
    e["params"] = sensitivity_params(sensitivity)
    m = adjust_outline(base, e)
    r = max(1, int(round(_px(m.shape, 4.0))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    if cleanup_min_area:
        a = max(cleanup_min_area, int(6.7e-4 * m.size))
        m = fill_holes(remove_small(m, a), max_area=int(1e-3 * m.size))
    return m


def _snap_edges(image, mask, band_at2000: float = 8.0, iterations: int = 3) -> np.ndarray:
    """Refine an outline found on the roughness image with colour, but only in a thin
    band (+-8 px per 2000 px) around it: the blurred roughness puts the edge a few px
    outside the rock, while colour edges are sharp locally (D-839)."""
    import cv2
    r = max(1, int(round(_px(mask.shape, band_at2000))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m8 = mask.astype(np.uint8)
    inner = cv2.erode(m8, k).astype(bool)
    outer = cv2.dilate(m8, k).astype(bool)
    lab = np.full(mask.shape, GC_BGD, np.uint8)
    lab[outer] = GC_PR_BGD
    lab[mask] = GC_PR_FGD
    lab[inner] = GC_FGD
    if not np.any(lab == GC_FGD):
        return mask
    vv, uu = np.nonzero(outer)
    v0, v1 = max(vv.min() - 2, 0), vv.max() + 3
    u0, u1 = max(uu.min() - 2, 0), uu.max() + 3
    sub = np.ascontiguousarray(np.asarray(image)[v0:v1, u0:u1, :3].astype(np.uint8))
    sl = lab[v0:v1, u0:u1].copy()
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(sub, sl, None, np.zeros((1, 65)), np.zeros((1, 65)), iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    out = mask.copy()
    out[v0:v1, u0:u1] = (sl == GC_FGD) | (sl == GC_PR_FGD)
    return out


def segment_view(image, view: View | None = None, method: str = "auto", rect=None, fg_strokes=None,
                 bg_strokes=None, init_labels=None, guide=None, iterations: int = 5,
                 sam_model: str = "mobile_sam", cleanup_min_area: int = 200, cue: str = "auto",
                 features=None, keep_on=None, strict: float = 1.0, sensitivity: float = 50.0,
                 edge=None) -> dict:
    """Automatic rock-mask suggestion for one photo.

    method: 'auto' (SAM if installed, else GrabCut), 'grabcut' or 'sam'.
    guide: optional bool image where rock can be (see :func:`guide_from_views`).
    cue: 'auto' (roughness for side views, colour for the top view -- seen from above
    the sand is rough too), 'texture' or 'colour'; used when the initialisation is
    computed here. features: optional 3-channel image GrabCut runs on instead of the
    photo. keep_on: optional bool image; only mask blobs touching it are kept.
    sensitivity: 0-100, 50 = default; higher finds more rock (:func:`sensitivity_params`).
    Returns {'mask': bool HxW, 'method': used, 'notes': [...]}."""
    notes = []
    if cue not in ("auto", "texture", "colour"):
        raise ValueError("cue must be 'auto', 'texture' or 'colour'")
    use_sam = False
    if method in ("auto", "sam"):
        from .models import SamSegmenter
        ok, why = SamSegmenter(sam_model).available()
        if ok:
            use_sam = True
        else:
            notes.append(f"SAM ({sam_model}) {why}; used GrabCut")
            if method == "sam":
                notes.append("requested 'sam' but it is not installed")
    elif method != "grabcut":
        raise ValueError("method must be 'auto', 'grabcut' or 'sam'")
    if init_labels is None and rect is None:
        if view is None:
            raise ValueError("need a calibrated view, a rect, or init labels for the automatic suggestion")
        use_tex = cue == "texture" or (cue == "auto" and view.kind != "top")
        ti = texture_init(view, image, guide=guide, strict=strict, sensitivity=sensitivity) if use_tex else None
        if ti is not None and ti["seed_fraction"] > 0:
            init_labels, features, keep_on, edge = ti["labels"], ti["features"], ti["floor"], ti["edge"]
            notes.append("initialised from the calibration and the rock's rough texture (blobs that reach "
                         "the sand)" + (" (guided by other views)" if guide is not None else ""))
        else:
            if ti is not None:
                notes.append("no rough blob reaches the sand: used the colour model instead")
            init_labels = auto_init_labels(view, image, guide=guide,
                                           colour_threshold=sensitivity_params(sensitivity)["colour_threshold"])
            notes.append("initialised from the calibration and a background colour model"
                         + (" (guided by other views)" if guide is not None else ""))
    if use_sam:
        from .models import SamSegmenter
        if rect is None:
            fgm = (init_labels == GC_PR_FGD) | (init_labels == GC_FGD)
            vv, uu = np.nonzero(fgm)
            rect = [uu.min(), vv.min(), uu.max(), vv.max()]
        fg_pts = [s[0] for s in fg_strokes or []]
        bg_pts = [s[0] for s in bg_strokes or []]
        mask = SamSegmenter(sam_model).predict(image, box=rect, fg_points=fg_pts, bg_points=bg_pts)
        used = sam_model
    else:
        mask = grabcut_segment(image if features is None else features, init_labels=init_labels, rect=rect,
                               fg_strokes=fg_strokes, bg_strokes=bg_strokes, iterations=iterations)
        used = "grabcut"
    if keep_on is not None:
        mask = keep_touching(mask, keep_on)
    if features is not None and used == "grabcut" and mask.any():
        mask = _snap_edges(image, mask)
    base = mask if (edge is not None and used == "grabcut") else None
    if base is not None:
        return {"mask": finish_texture_mask(base, edge, sensitivity, cleanup_min_area), "method": used,
                "notes": notes + ([f"sensitivity {sensitivity:g} (50 = default)"] if sensitivity != 50 else []),
                "base": base, "edge": edge}
    if cleanup_min_area:
        if features is None:
            mask = fill_holes(remove_small(mask, cleanup_min_area), max_area=cleanup_min_area)
        else:      # texture cue: open with a 4-px disk (per 2000 px) to drop thin strips along
            # glass/sand edges; area limits scale with the photo (2000 / 3000 px at 3 MP)
            import cv2
            r = max(1, int(round(_px(mask.shape, 4.0))))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
            a = max(cleanup_min_area, int(6.7e-4 * mask.size))
            mask = fill_holes(remove_small(mask, a), max_area=int(1e-3 * mask.size))
    return {"mask": mask, "method": used, "notes": notes}


# ----------------------------------------------------------------------
# mask editing (pure: input arrays are never modified)
# ----------------------------------------------------------------------
def brush(mask, stroke, radius: float, mode: str = "add") -> np.ndarray:
    """Paint a polyline stroke [[u, v], ...] of the given radius (px).
    mode 'add' sets True, 'subtract' sets False."""
    import cv2
    m = np.asarray(mask, bool)
    paint = np.zeros(m.shape, np.uint8)
    pts = np.round(np.atleast_2d(np.asarray(stroke, float)) * 16).astype(np.int32)
    r = max(1, int(round(radius)))
    for p in pts:
        cv2.circle(paint, (int(p[0]), int(p[1])), r * 16, 1, -1, lineType=cv2.LINE_8, shift=4)
    if len(pts) > 1:
        cv2.polylines(paint, [pts], False, 1, thickness=2 * r + 1, lineType=cv2.LINE_8, shift=4)
    return _apply(m, paint.astype(bool), mode)


def polygon(mask, vertices, mode: str = "add") -> np.ndarray:
    """Fill a polygon [[u, v], ...] (>= 3 vertices) into the mask (add / subtract)."""
    import cv2
    m = np.asarray(mask, bool)
    v = np.atleast_2d(np.asarray(vertices, float))
    if len(v) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    paint = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(paint, [np.round(v * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return _apply(m, paint.astype(bool), mode)


def _apply(m, region, mode):
    if mode == "add":
        return m | region
    if mode == "subtract":
        return m & ~region
    raise ValueError("mode must be 'add' or 'subtract'")


def remove_small(mask, min_area: int) -> np.ndarray:
    """Drop connected rock components smaller than min_area pixels."""
    from scipy import ndimage
    m = np.asarray(mask, bool)
    lab, n = ndimage.label(m)
    if n == 0:
        return m.copy()
    area = np.bincount(lab.ravel())
    keep = area >= min_area
    keep[0] = False
    return keep[lab]


def fill_holes(mask, max_area: int | None = None) -> np.ndarray:
    """Fill enclosed background holes (all, or only those up to max_area px)."""
    from scipy import ndimage
    m = np.asarray(mask, bool)
    holes = ndimage.binary_fill_holes(m) & ~m
    if max_area is not None:
        lab, n = ndimage.label(holes)
        if n:
            area = np.bincount(lab.ravel())
            small = area <= max_area
            small[0] = False
            holes = small[lab]
    return m | holes


def smooth(mask, radius: int = 2) -> np.ndarray:
    """Morphological open + close with a disk of the given radius (px)."""
    import cv2
    m = np.asarray(mask, bool).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def mask_iou(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else 1.0


def mask_to_png(mask) -> bytes:
    """Bool mask -> 8-bit PNG bytes (0 / 255), for the UI."""
    import cv2
    ok, buf = cv2.imencode(".png", np.asarray(mask, bool).astype(np.uint8) * 255)
    if not ok:
        raise IOError("PNG encode failed")
    return buf.tobytes()


def mask_from_png(data: bytes) -> np.ndarray:
    import cv2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("PNG decode failed")
    return img > 127


def load_image(path) -> np.ndarray:
    """Read a photo as RGB uint8 (EXIF orientation applied by OpenCV)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path, rgb) -> None:
    import cv2
    if not cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR)):
        raise IOError(f"cannot write {path}")


def _segment_with_retry(image, view, method, guide, thr, retries=3, collapse=0.85, cue="auto",
                        sensitivity: float = 50.0):
    """GrabCut from the automatic init; if the result swallows > ``collapse`` of the
    candidate region (GrabCut found no separation), retry stricter: the colour
    threshold x1.4, or for the texture cue the roughness threshold x1.25."""
    notes = []
    use_tex = cue == "texture" or (cue == "auto" and view.kind != "top")
    strict = 1.0
    for i in range(retries + 1):
        ti = texture_init(view, image, guide=guide, strict=strict, sensitivity=sensitivity) if use_tex else None
        if ti is not None and ti["seed_fraction"] > 0:
            lab = ti["labels"]
            res = segment_view(image, view, method=method, init_labels=lab, features=ti["features"],
                               keep_on=ti["floor"], edge=ti["edge"], sensitivity=sensitivity)
            res["notes"].append("initialised from the rock's rough texture (blobs that reach the sand)")
        else:
            if ti is not None:
                notes.append("no rough blob reaches the sand: used the colour model")
                use_tex = False
            lab = auto_init_labels(view, image, guide=guide, colour_threshold=thr)
            res = segment_view(image, view, method=method, init_labels=lab)
        cand = view_region_masks(view)["rockable"] if guide is None else (guide & (lab != GC_BGD))
        frac = np.count_nonzero(res["mask"] & cand) / max(np.count_nonzero(cand), 1)
        if frac <= collapse or res["method"] != "grabcut":
            break
        if use_tex:
            notes.append(f"roughness threshold x{strict:.2f}: mask covered {frac:.0%} of the candidate "
                         "region; retrying stricter")
            strict *= 1.25
        else:
            notes.append(f"colour threshold {thr:.1f}: mask covered {frac:.0%} of the candidate region "
                         "(no separation); retrying stricter")
            thr *= 1.4
    res["notes"] = notes + res["notes"] + ([f"roughness threshold x{strict:.2f}"] if use_tex
                                           else [f"colour threshold {thr:.1f} Delta-E"])
    return res


def auto_segment_views(images: dict, views: dict, method: str = "auto", colour_threshold: float = 14.0,
                       guide_dx: float = 0.01, cue: str = "auto", sensitivity: float = 50.0) -> dict:
    """Automatic suggestion for a whole photo set (dicts keyed by view name).

    The top view is segmented first from the calibration alone (rock over sand is
    the easiest contrast); its prism is reprojected into every other view as a
    guide (:func:`guide_from_views`). Without a top view each view is segmented
    on its own. Returns {name: segment_view result}."""
    out = {}
    colour_threshold = colour_threshold * sensitivity_params(sensitivity)["colour_threshold"] / 14.0
    tops = [k for k, v in views.items() if v.kind == "top"]
    for k in tops:
        out[k] = _segment_with_retry(images[k], views[k], method, None, colour_threshold, cue=cue, sensitivity=sensitivity)
        out[k]["notes"].append("unguided (top view first)")
    for k, v in views.items():
        if k in out:
            continue
        g = None
        if tops:
            g = guide_from_views(v, [views[t] for t in tops], [out[t]["mask"] for t in tops], dx=guide_dx)
        out[k] = _segment_with_retry(images[k], v, method, g, colour_threshold, cue=cue, sensitivity=sensitivity)
        out[k]["notes"].append("guided by the top-view prism" if tops else "unguided (no top view)")
    return out
