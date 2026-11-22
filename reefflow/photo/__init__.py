"""Phase 6 -- rockscape reconstruction from photos (spec Feature 1).

Workflow (each step is a pure function a UI backend can call):

1. Tank geometry           :class:`TankGeometry` (from a project file or numbers)
2. Calibration per photo   :func:`calibrate_view` -- user clicks named tank points
                           (:func:`recommended_points`); solves focal length + full
                           pose by PnP with flat-interface refraction (PLAN C9)
3. Segmentation per photo  :func:`auto_segment_views` / :func:`segment_view`
                           (GrabCut, or SAM if installed) + mask edits
                           (:func:`brush`, :func:`polygon`, ...)
4. Carving                 :func:`carve_visual_hull` on a :class:`VoxelGrid`
5. Cleanup / edits         :func:`apply_edits` (tunnels, arches, open regions, add)
6. Check                   :func:`side_by_side` (reconstruction vs photo)
7. Export / report         :func:`export_stl`, :func:`volume_report`

:func:`reconstruct` chains 2-7. See DECISIONS.md and ASSUMPTIONS.md (Phase 6) for decisions,
assumptions and the license audit; validation: ``python -m reefflow.photo.validation``.
"""
from .calibration import CalibrationError, CalibrationResult, calibrate_view, check_tank_size, focal_px_from_35mm
from .camera import Camera, View
from .carve import (DEFAULT_BULK_DENSITY, CarveResult, VoxelGrid, carve_visual_hull, compare_occupancy,
                    export_stl, occupancy_to_mesh, remove_small_components, rock_weight, volume_report)
from .compare import overlay, reproject_silhouette, side_by_side
from .edit import (add_material, apply_edit, apply_edits, carve_arch, carve_box, carve_sphere, carve_tunnel,
                   mark_open, region)
from .models import (ModelNotInstalled, SamSegmenter, capabilities, download_model, estimate_depth,
                     model_status, photogrammetry_reconstruct)
from .pipeline import reconstruct
from .segment import (auto_init_labels, auto_segment_views, brush, fill_holes, grabcut_segment, guide_from_views,
                      load_image, mask_from_png, mask_iou, mask_to_png, polygon, remove_small, save_image,
                      segment_view, smooth)
from .tankref import TankGeometry, recommended_points

__all__ = [n for n in dir() if not n.startswith("_")]
