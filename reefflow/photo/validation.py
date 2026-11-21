"""Phase-6 validation: reconstruct a KNOWN rockscape from synthetic photos.

Run:  python -m reefflow.photo.validation  [--quick]
Out:  verification/results/10_photo/results.json + PNG figures

Set-up
------
* Tank: the demo 75-gal project (47 x 17 x 20.5 in internal, 1/2 in glass, water
  1 in below the rim, 2 in sand), read from the project file written by
  ``reefflow.demo.write_demo`` into a temp dir. Rock: that project's rock.stl
  (two islands), loaded with ``geometry.load_mesh`` like any user STL.
* Ground truth: ``geometry.voxelize_mesh`` of the STL on the reconstruction grid
  (x in [0, L], y in [0, W], z from sand to water surface), dx = 5 mm (the
  'standard' CFD preset).
* Photos: top (1.05 m above the rim), front (1.45 m in front of the glass,
  crouching height), left side (0.8 m from the side glass); 1600 x 1200 px,
  f = 1200 px (~67 deg horizontal FOV, phone-like), slight offsets/tilt/roll;
  rendered through the refraction model (n_water 1.34, n_glass 1.5) by
  ``photo.synthetic``. Render seed 7 is held out (the GrabCut colour threshold
  was chosen on seeds 1-3).
* Clicks: the 8 recommended reference points per view, exact projections plus
  isotropic Gaussian noise sigma (px) per coordinate.

Acceptance criteria (declared before the first run, DECISIONS D-110 / D-607)
---------------------------------------------------------------------------
Calibration
  C1  sigma = 0: refractive PnP recovers the camera exactly: residual RMS < 1e-6 px,
      camera-centre error < 1e-6 m, focal error < 1e-6 relative (model consistency).
  C2  sigma = 2 px (8 points, 7 unknowns): mean residual RMS <= 3 px over trials
      (expected sqrt(2) sigma sqrt(1 - 7/16) = 2.1 px), mean camera-centre error
      <= 25 mm and mean |focal error| <= 2 % (development runs gave 1-22 mm).
Carving (IoU, recall = share of true rock voxels kept, volume error = V/V_true - 1)
  V1  exact cameras + ground-truth masks, 3 views: recall >= 0.98, IoU >= 0.70,
      volume error in [0, +45 %].
      Basis: the visual hull of a sphere from 3 orthogonal orthographic views is the
      Steinmetz tricylinder, 8(2 - sqrt 2) r^3 = 1.119 x sphere (IoU 0.89); from 2
      views 1.27 x (IoU 0.79). Here the left view sees both islands superimposed, so
      the left island is constrained essentially by 2 views; perspective rays and
      surface-roughness concavities (noise amplitude ~25 % of the blob radius) add
      overfill. Hence IoU >= 0.70 and overfill <= 45 %. Recall < 1 only from
      pixel/voxel point sampling; tolerance 1 px.
  V2  calibration from sigma = 2 px clicks + ground-truth masks (8 trials):
      mean IoU >= IoU(V1) - 0.05, mean recall >= 0.95.
  V3  full pipeline: sigma = 2 px clicks + automatic GrabCut masks (held-out seed):
      every view's mask IoU >= 0.90 vs the true mask; voxel IoU >= 0.65;
      volume error in [0, +50 %].
  V4  click-noise sensitivity, sigma in {0, 1, 2, 4, 8} px, 6 trials each (GT masks):
      mean IoU at sigma = 4 px >= IoU(V1) - 0.10. sigma = 8 px reported only.
  V5  refraction ablation (reported, no pass/fail): (a) true cameras but projection
      without refraction; (b) calibration AND carving with the pinhole model.
  V6  STL export: voxelize_mesh(exported STL) on the same grid reproduces the
      occupancy, mismatch <= 0.1 % of occupied voxels.
  V1c (informational; added after the first run showed V1 failing, NOT a criterion)
      exact cameras + true masks with a 4th photo through the front glass from 40 deg
      to the left, and the per-island breakdown -- to show where the overfill comes
      from (diagonal 'corner wedges' unconstrained by axis-aligned views).
  V7  edit tools: a blind cave (invisible to every silhouette) cut into the truth is
      overfilled by the hull; carving the same capsule with edit.carve_tunnel must
      raise IoU vs that truth back to within 0.005 of IoU(V1 hull vs original truth).
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

import reefflow  # noqa: F401  (offline Taichi guard; not needed here but project convention)

from .. import demo, geometry as G
from . import segment as SG
from . import synthetic as SY
from .calibration import calibrate_view
from .camera import View
from .carve import VoxelGrid, carve_visual_hull, compare_occupancy, export_stl, volume_report
from .compare import side_by_side
from .edit import carve_tunnel
from .tankref import TankGeometry, recommended_points

OUT = Path(__file__).resolve().parents[2] / "verification" / "results" / "10_photo"
DX = 0.005
HOLDOUT_SEED = 7
SIGMAS = (0.0, 1.0, 2.0, 4.0, 8.0)
C_BLUE, C_ORANGE, C_AQUA, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"


def setup(tmp):
    proj = demo.write_demo(tmp)
    tank = TankGeometry.from_project(proj)
    v, f = G.load_mesh(Path(tmp) / "rock.stl", unit="mm")
    grid = VoxelGrid.for_tank(tank, DX)
    truth = G.voxelize_mesh(v, f, grid.origin, grid.dx, grid.shape)
    cams = SY.default_cameras(tank)
    true_views = {k: View(k, c, tank) for k, c in cams.items()}
    return tank, v, f, grid, truth, true_views


def noisy_clicks(view, sigma, rng):
    names = [p["name"] for p in recommended_points(view.kind)]
    exact = SY.true_clicks(view, names)
    return {n: (np.asarray(u) + rng.normal(0, sigma, 2)).tolist() for n, u in exact.items()}, exact


def calibrate_all(true_views, tank, sigma, rng, refraction=True):
    views, cals = {}, {}
    for k, tv in true_views.items():
        clicks, _ = noisy_clicks(tv, sigma, rng)
        cal = calibrate_view(k, (tv.camera.width, tv.camera.height), tank, clicks, refraction=refraction)
        views[k], cals[k] = cal.view, cal
    return views, cals


def cal_errors(cals, true_views):
    out = {}
    for k, c in cals.items():
        tc = true_views[k].camera
        out[k] = {"rms_px": c.rms_px, "max_px": c.max_px,
                  "centre_err_mm": 1e3 * float(np.linalg.norm(c.view.camera.centre - tc.centre)),
                  "focal_err": c.view.camera.f / tc.f - 1, "centre_std_mm": c.centre_std_mm,
                  "pinhole_only_rms_px": c.pinhole_rms_px}
    return out


def carve(views, masks, grid, truth, tol=1):
    ks = list(views)
    r = carve_visual_hull([views[k] for k in ks], [masks[k] for k in ks], grid, tolerance_px=tol)
    return r.occupancy, compare_occupancy(r.occupancy, truth), r.stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer trials (smoke run)")
    a = ap.parse_args(argv)
    n_trials = 2 if a.quick else 6
    n_v2 = 2 if a.quick else 8
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    res = {"dx_m": DX, "image_size": [1600, 1200], "f_px": 1200.0, "holdout_seed": HOLDOUT_SEED}
    with tempfile.TemporaryDirectory() as tmp:
        tank, rv, rf, grid, truth, true_views = setup(tmp)
    res["tank"] = tank.to_dict()
    res["grid"] = grid.to_dict()
    vt = float(truth.sum()) * grid.cell_volume
    res["truth"] = {"voxels": int(truth.sum()), "volume_L": vt * 1e3}
    print(f"truth: {truth.sum()} voxels, {vt * 1e3:.2f} L above the sand; grid {grid.shape}")

    images, gt_masks = {}, {}
    for k, v in true_views.items():
        images[k], gt_masks[k] = SY.render_view(v, rv, rf, seed=HOLDOUT_SEED)
        SG.save_image(OUT / f"photo_{k}.png", images[k])
    # refraction displacement of rock surface points, if uncorrected (pinhole vs refractive)
    disp = {}
    for k, v in true_views.items():
        P = rv[rv[:, 2] > tank.sand_depth][::5]
        uv_r, ok = v.project(P)
        uv_p, _ = v.camera.project_points_pinhole(P)
        d = np.linalg.norm(uv_r - uv_p, axis=1)[ok]
        disp[k] = {"mean_px": float(d.mean()), "max_px": float(d.max())}
    res["refraction_displacement_rock_px"] = disp

    # ---- C1 -------------------------------------------------------------
    rng = np.random.default_rng(1)
    _, cals0 = calibrate_all(true_views, tank, 0.0, rng)
    e0 = cal_errors(cals0, true_views)
    c1 = all(e["rms_px"] < 1e-6 and e["centre_err_mm"] < 1e-3 and abs(e["focal_err"]) < 1e-6 for e in e0.values())
    res["C1"] = {"errors": e0, "pass": bool(c1)}
    print("C1", c1, {k: (e["rms_px"], e["centre_err_mm"]) for k, e in e0.items()})

    # ---- V1 -------------------------------------------------------------
    t0 = time.time()
    occ1, m1, st1 = carve(true_views, gt_masks, grid, truth)
    t_carve = time.time() - t0
    v1 = m1["recall"] >= 0.98 and m1["iou"] >= 0.70 and 0 <= m1["volume_error"] <= 0.45
    res["V1"] = {"metrics": m1, "carve_stats": st1, "carve_time_s": t_carve, "pass": bool(v1)}
    print("V1", v1, m1, f"{t_carve:.1f}s")
    # 2-view information value (top + front only)
    _, m1b, _ = carve({k: true_views[k] for k in ("top", "front")}, gt_masks, grid, truth)
    res["V1_two_views_top_front"] = m1b
    # V1c informational: per island + a 4th, oblique front photo
    xs = grid.origin[0] + grid.dx * np.arange(grid.shape[0])
    isl = {"left_island": xs < 0.6, "right_island": xs >= 0.6}
    res["V1_per_island"] = {n: compare_occupancy(occ1[sel], truth[sel]) for n, sel in isl.items()}
    c0 = true_views["front"].camera
    a = np.radians(40.0)
    tgt = np.array([tank.length / 2, tank.width / 2, 0.24])
    eye = tgt + 1.6 * np.array([-np.sin(a), -np.cos(a), 0.0]) + [0, 0, 0.06]
    obl = View("front", type(c0).look_at(eye, tgt, c0.f, c0.width, c0.height), tank)
    v4 = dict(true_views, front_oblique=obl)
    m4 = dict(gt_masks, front_oblique=SY.render_mask(obl, rv, rf))
    occ4, m1c, _ = carve(v4, m4, grid, truth)
    res["V1c_four_views_with_oblique"] = {"metrics": m1c, "camera_centre_m": obl.camera.centre.tolist(),
                                          "per_island": {n: compare_occupancy(occ4[sel], truth[sel])
                                                         for n, sel in isl.items()}}
    print("V1c", m1c)

    # ---- V2 + C2 ----------------------------------------------------------
    rng = np.random.default_rng(2)
    rows, calrows = [], []
    for _ in range(n_v2):
        views, cals = calibrate_all(true_views, tank, 2.0, rng)
        _, m, _ = carve(views, gt_masks, grid, truth)
        rows.append(m); calrows.append(cal_errors(cals, true_views))
    iou2 = float(np.mean([r["iou"] for r in rows])); rec2 = float(np.mean([r["recall"] for r in rows]))
    v2 = iou2 >= m1["iou"] - 0.05 and rec2 >= 0.95
    res["V2"] = {"trials": rows, "mean_iou": iou2, "mean_recall": rec2,
                 "mean_volume_error": float(np.mean([r["volume_error"] for r in rows])), "pass": bool(v2)}
    allc = [c for row in calrows for c in row.values()]
    rms2 = float(np.mean([c["rms_px"] for c in allc]))
    ce2 = float(np.mean([c["centre_err_mm"] for c in allc]))
    fe2 = float(np.mean([abs(c["focal_err"]) for c in allc]))
    per_view = {k: {"mean_rms_px": float(np.mean([r[k]["rms_px"] for r in calrows])),
                    "mean_centre_err_mm": float(np.mean([r[k]["centre_err_mm"] for r in calrows])),
                    "mean_reported_centre_std_mm": float(np.mean([np.linalg.norm(r[k]["centre_std_mm"]) for r in calrows])),
                    "mean_abs_focal_err": float(np.mean([abs(r[k]["focal_err"]) for r in calrows])),
                    "mean_pinhole_only_rms_px": float(np.mean([r[k]["pinhole_only_rms_px"] for r in calrows]))}
                for k in true_views}
    c2 = rms2 <= 3.0 and ce2 <= 25.0 and fe2 <= 0.02
    res["C2"] = {"mean_rms_px": rms2, "mean_centre_err_mm": ce2, "mean_abs_focal_err": fe2,
                 "per_view": per_view, "pass": bool(c2)}
    print("C2", c2, rms2, ce2, fe2, "| V2", v2, iou2, rec2)

    # ---- V3 full pipeline -------------------------------------------------
    rng = np.random.default_rng(3)
    views3, cals3 = calibrate_all(true_views, tank, 2.0, rng)
    t0 = time.time()
    seg = SG.auto_segment_views(images, views3, method="grabcut")
    t_seg = time.time() - t0
    masks3 = {k: s["mask"] for k, s in seg.items()}
    miou = {k: SG.mask_iou(masks3[k], gt_masks[k]) for k in masks3}
    occ3, m3, st3 = carve(views3, masks3, grid, truth)
    v3 = min(miou.values()) >= 0.90 and m3["iou"] >= 0.65 and 0 <= m3["volume_error"] <= 0.50
    vol3 = volume_report(occ3, grid)
    res["V3"] = {"mask_iou": miou, "segmentation_notes": {k: s["notes"] for k, s in seg.items()},
                 "segmentation_time_s": t_seg, "metrics": m3, "calibration": cal_errors(cals3, true_views),
                 "volume_report": vol3, "truth_weight_kg_at_default_density": vt * 1400.0, "pass": bool(v3)}
    print("V3", v3, miou, m3)
    sil = {}
    for k in true_views:
        sbs = side_by_side(images[k], views3[k], occ3, grid, masks3[k], label=k)
        SG.save_image(OUT / f"compare_{k}.png", sbs["image"])
        sil[k] = sbs["silhouette_iou"]
    res["V3"]["reprojected_silhouette_iou"] = sil

    # ---- V4 sensitivity ---------------------------------------------------
    rng = np.random.default_rng(4)
    sens = {}
    for s in SIGMAS:
        rows, cerr = [], []
        for _ in range(n_trials if s > 0 else 1):
            views, cals = calibrate_all(true_views, tank, s, rng)
            _, m, _ = carve(views, gt_masks, grid, truth)
            rows.append(m)
            cerr += [c["centre_err_mm"] for c in cal_errors(cals, true_views).values()]
        sens[str(s)] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))]
                        for k in ("iou", "recall", "volume_error", "precision")}
        sens[str(s)]["centre_err_mm"] = [float(np.mean(cerr)), float(np.std(cerr))]
        print("V4 sigma", s, sens[str(s)])
    ok4 = sens["4.0"]["iou"][0] >= m1["iou"] - 0.10
    res["V4"] = {"by_sigma_mean_std": sens, "trials": n_trials, "pass": bool(ok4)}

    # ---- V5 ablation --------------------------------------------------------
    nr = {k: View(k, v.camera, tank, refraction=False) for k, v in true_views.items()}
    _, m5a, _ = carve(nr, gt_masks, grid, truth)
    rng = np.random.default_rng(5)
    views5, cals5 = calibrate_all(true_views, tank, 0.0, rng, refraction=False)
    _, m5b, _ = carve(views5, gt_masks, grid, truth)
    res["V5"] = {"true_cameras_no_refraction": m5a, "pinhole_calibrated_and_carved": m5b,
                 "pinhole_calibration": cal_errors(cals5, true_views)}
    print("V5", m5a, m5b)

    # ---- V6 STL round trip ----------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        ex = export_stl(occ3, grid, Path(tmp) / "rock_photo.stl")
        v6v, v6f = G.load_mesh(Path(tmp) / "rock_photo.stl", unit="mm")
    back = G.voxelize_mesh(v6v, v6f, grid.origin, grid.dx, grid.shape)
    mism = int(np.count_nonzero(back ^ occ3))
    v6 = mism <= 1e-3 * occ3.sum()
    res["V6"] = {"mismatch_voxels": mism, "occupied": int(occ3.sum()), "mesh": ex["mesh"], "pass": bool(v6)}
    print("V6", v6, mism)

    # ---- V7 edit tools ----------------------------------------------------
    cave = {"points": [[0.33, 0.10, tank.sand_depth + 0.06], [0.33, 0.20, tank.sand_depth + 0.06]], "radius": 0.025}
    truth_cave = carve_tunnel(truth, grid, cave["points"], cave["radius"])
    before = compare_occupancy(occ1, truth_cave)
    after = compare_occupancy(carve_tunnel(occ1, grid, cave["points"], cave["radius"]), truth_cave)
    # the cave must be blind: invisible in every true silhouette
    cave_mesh = SY.orient_outward(*G_mesh(truth_cave, grid))
    sil_change = {}
    for k, v in true_views.items():
        _, mk = SY.render_view(v, *cave_mesh, seed=0, clip=False)
        sil_change[k] = 1 - SG.mask_iou(mk, SY.render_view(v, *SY.orient_outward(*G_mesh(truth, grid)), seed=0,
                                                         clip=False)[1])
    v7 = after["iou"] >= m1["iou"] - 0.005 and after["iou"] > before["iou"]
    res["V7"] = {"cave": cave, "cave_volume_L": float((truth ^ truth_cave).sum()) * grid.cell_volume * 1e3,
                 "hull_vs_cave_truth": before, "after_carve_tunnel": after,
                 "silhouette_change_from_cave": sil_change, "pass": bool(v7)}
    print("V7", v7, before["iou"], after["iou"], sil_change)

    res["all_pass"] = bool(all(res[c]["pass"] for c in ("C1", "C2", "V1", "V2", "V3", "V4", "V6", "V7")))
    res["runtime_s"] = time.time() - t_start
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    figures(res, images, gt_masks, masks3, true_views, cals3, truth, occ1, occ3, grid)
    print("ALL PASS" if res["all_pass"] else "SOME CRITERIA FAILED", f"({res['runtime_s']:.0f} s)")
    return res


def G_mesh(occ, grid):
    from .carve import occupancy_to_mesh
    return occupancy_to_mesh(occ, grid, extend_below=0)


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def figures(res, images, gt_masks, masks3, true_views, cals3, truth, occ1, occ3, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888", "axes.labelcolor": "#333",
                         "xtick.color": "#555", "ytick.color": "#555"})
    ks = list(true_views)
    # 1. photos with clicks
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, k in zip(axs, ks):
        c = cals3[k]
        ax.imshow(images[k])
        ax.scatter(c.uv_clicked[:, 0], c.uv_clicked[:, 1], s=60, marker="+", c=C_ORANGE, linewidths=1.5,
                   label="clicked (sigma 2 px)")
        ax.scatter(c.uv_model[:, 0], c.uv_model[:, 1], s=40, facecolors="none", edgecolors=C_AQUA,
                   linewidths=1.5, label="refractive model")
        ax.set_title(f"{k}: RMS residual {c.rms_px:.2f} px", fontsize=10)
        ax.axis("off")
    axs[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle("Synthetic photos (held-out render seed) with calibration clicks", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig1_photos_calibration.png", dpi=110); plt.close(fig)

    # 2. masks
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, k in zip(axs, ks):
        img = images[k].astype(float) / 255 * 0.55
        a, g = masks3[k], gt_masks[k]
        rgb = img.copy()
        rgb[a & g] = 0.55 * rgb[a & g] + 0.45 * np.array([0.6, 0.6, 0.58])
        rgb[a & ~g] = np.array([0xeb, 0x68, 0x34]) / 255
        rgb[~a & g] = np.array([0x2a, 0x78, 0xd6]) / 255
        ax.imshow(rgb); ax.axis("off")
        ax.set_title(f"{k}: GrabCut mask IoU {res['V3']['mask_iou'][k]:.3f}", fontsize=10)
    fig.suptitle("Automatic segmentation vs true mask   (grey = agree, orange = extra, blue = missed)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig2_segmentation.png", dpi=110); plt.close(fig)

    # 3. voxel error maps: plan (count along z) and elevation (count along y)
    cmap = ListedColormap(["#ffffff", "#c9c8c2", C_ORANGE, C_BLUE])

    def classes(rec, proj_axis):
        tp = (rec & truth).any(proj_axis)
        fp = (rec & ~truth).any(proj_axis)
        fn = (~rec & truth).any(proj_axis)
        c = np.zeros(tp.shape, int)
        c[tp] = 1; c[fp & ~tp] = 2; c[fn] = 3
        return c

    def slab(rec, k):
        c = np.zeros(truth.shape[:2], int)
        t, r = truth[:, :, k], rec[:, :, k]
        c[t & r] = 1; c[r & ~t] = 2; c[t & ~r] = 3
        return c

    kmid = int((0.10) / grid.dx)
    fig, axs = plt.subplots(2, 2, figsize=(14, 6.5))
    ext_xy = [0, grid.shape[0] * grid.dx * 100, 0, grid.shape[1] * grid.dx * 100]
    for col, (rec, name) in enumerate(((occ1, "V1 exact cameras + true masks"), (occ3, "V3 full pipeline"))):
        m = res["V1"]["metrics"] if col == 0 else res["V3"]["metrics"]
        axs[0, col].imshow(slab(rec, kmid).T, origin="lower", cmap=cmap, vmin=0, vmax=3, extent=ext_xy,
                           interpolation="nearest")
        axs[0, col].set_title(f"{name}: plan slice z = sand + {kmid * grid.dx * 100:.1f} cm   "
                              f"(IoU {m['iou']:.3f}, vol err {m['volume_error']:+.1%})", fontsize=9)
        axs[0, col].set_xlabel("x [cm]"); axs[0, col].set_ylabel("y [cm]")
        jmid = int(0.25 / grid.dx)
        c = np.zeros((grid.shape[0], grid.shape[2]), int)
        t, r = truth[:, jmid, :], rec[:, jmid, :]
        c[t & r] = 1; c[r & ~t] = 2; c[t & ~r] = 3
        axs[1, col].imshow(c.T, origin="lower", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
                           extent=[0, grid.shape[0] * grid.dx * 100, 0, grid.shape[2] * grid.dx * 100])
        axs[1, col].set_title(f"elevation slice y = {jmid * grid.dx * 100:.1f} cm", fontsize=9)
        axs[1, col].set_xlabel("x [cm]"); axs[1, col].set_ylabel("z above sand [cm]")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#c9c8c2", label="rock in both"), Patch(color=C_ORANGE, label="overfill (recon only)"),
                        Patch(color=C_BLUE, label="missed (truth only)")], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "fig3_voxel_slices.png", dpi=110); plt.close(fig)

    # 4. sensitivity
    sens = res["V4"]["by_sigma_mean_std"]
    s = np.array([float(x) for x in sens])
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, col, lab in (("iou", C_BLUE, "IoU"), ("recall", C_AQUA, "recall")):
        mu = np.array([sens[x][key][0] for x in sens]); sd = np.array([sens[x][key][1] for x in sens])
        axs[0].errorbar(s, mu, yerr=sd, color=col, marker="o", ms=6, lw=2, capsize=3, label=lab)
    axs[0].axhline(res["V1"]["metrics"]["iou"] - 0.10, color=C_GRAY, ls="--", lw=1)
    axs[0].text(0.1, res["V1"]["metrics"]["iou"] - 0.095, "V4 criterion (IoU at 4 px)", color="#555", fontsize=8)
    axs[0].set_xlabel("click noise sigma [px]"); axs[0].set_ylabel("voxel metric (true masks)")
    axs[0].legend(frameon=False); axs[0].grid(alpha=0.25); axs[0].set_title("Reconstruction vs click noise", fontsize=10)
    mu = np.array([sens[x]["centre_err_mm"][0] for x in sens]); sd = np.array([sens[x]["centre_err_mm"][1] for x in sens])
    axs[1].errorbar(s, mu, yerr=sd, color=C_ORANGE, marker="o", ms=6, lw=2, capsize=3)
    axs[1].set_xlabel("click noise sigma [px]"); axs[1].set_ylabel("camera-centre error [mm]")
    axs[1].grid(alpha=0.25); axs[1].set_title("Calibration error vs click noise (all 3 views)", fontsize=10)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig4_noise_sensitivity.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
