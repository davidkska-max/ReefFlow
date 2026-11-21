# LICENSES.md — third-party software and models

Rule (from the project spec): every library and model must permit commercial use. Models with non-commercial licenses are excluded.

| Component | Version | License | Commercial use | Used for |
|---|---|---|---|---|
| Python (CPython) | 3.12 | PSF-2.0 | Yes | runtime |
| Taichi | 1.7.4 | Apache-2.0 | Yes | GPU/CPU kernels |
| NumPy | ≥ 1.26 | BSD-3-Clause | Yes | arrays |
| SciPy | ≥ 1.11 | BSD-3-Clause | Yes | numerics, connected-component labelling |
| Matplotlib | ≥ 3.8 | Matplotlib License (PSF-based, BSD-compatible) | Yes | plots |
| PyYAML | ≥ 6 | MIT | Yes | project files |
| pytest (dev only) | ≥ 8 | MIT | Yes | tests |
| trimesh | 5.2 (≥ 4) | MIT | Yes | STL/OBJ/PLY rock import and export |
| scikit-image | 0.26 (≥ 0.22) | BSD-3-Clause | Yes | marching cubes (procedural rock meshes) |
| networkx | 3.7 | BSD-3-Clause | Yes | scikit-image dependency |
| imageio | 2.38 | BSD-2-Clause | Yes | scikit-image dependency |
| tifffile | 2026.10 | BSD-3-Clause | Yes | scikit-image dependency |
| lazy_loader | 0.6 | BSD-3-Clause | Yes | scikit-image dependency |
| Pillow | 13.0 | MIT-CMU (HPND) | Yes | matplotlib / scikit-image dependency |
| pyamg | 5.4 | MIT | Yes | algebraic multigrid (water-age and projection solves) |
| segno | 1.6.6 | BSD-3-Clause | Yes | QR code for the phone photo link (pure Python, no dependencies) |
| opencv-python-headless | 5.0.0.93 (≥ 4.8) | Apache-2.0 | Yes | GrabCut segmentation, SQPnP pose, rasterisation, PNG I/O (photo reconstruction) |

Notes on the OpenCV wheel (`LICENSE-3RD-PARTY.txt` in the dist-info):
- The Windows/Linux wheels bundle **FFmpeg** (`opencv_videoio_ffmpeg*.dll`) under **LGPL-2.1**. It is dynamically loaded and used only for video I/O, which ReefFlow does not call.
- LGPL permits commercial redistribution of an unmodified, dynamically linked library, provided the license text is shipped and users can replace the library. The wheel already ships the text. Record this in LICENSES.md.
- Other bundled codecs and image libraries (libpng, zlib, libjpeg-turbo, libwebp, openjpeg, libvpx, dav1d, …) are BSD, MIT, zlib or similar permissive licenses.
- The *headless* wheel contains no Qt (Qt is LGPL and ships only in the non-headless Linux/macOS wheels).

## Optional models (Phase 6) (not installed; import-guarded; user-initiated download only)

| Model / package | Version | License (code / weights) | Commercial use | Status in ReefFlow |
|---|---|---|---|---|
| MobileSAM (`mobile_sam`, weights `mobile_sam.pt`, ~39 MB) | repo master (2023) | Apache-2.0 / Apache-2.0 | Yes | Optional segmentation (`SamSegmenter('mobile_sam')`) |
| Segment Anything ViT-B (`segment_anything`, `sam_vit_b_01ec64.pth`, ~375 MB) | 1.0 | Apache-2.0 / Apache-2.0 | Yes | Optional segmentation |
| Depth Anything V2 **Small** (`depth_anything_v2_vits.pth`, ~99 MB) | V2 | Apache-2.0 / Apache-2.0 | Yes | Listed; depth interface is a stub (spec item 6) |
| Depth Anything V2 **Base / Large / Giant** | V2 | CC-BY-NC-4.0 | **No** | **Excluded.** Registry entry exists only so the UI can say why; `download_model` refuses |
| PyTorch (CPU build), needed by any model above | ≥ 2.x | BSD-3-Clause | Yes | Not a dependency; installed by the user only to enable SAM |
| timm (MobileSAM dependency) | ≥ 0.9 | Apache-2.0 | Yes | Only with MobileSAM |

Candidates for a later audit: SAM 2.1 hiera-tiny (Apache-2.0).

Weight files are cached in `%LOCALAPPDATA%/ReefFlow/models`, or `$REEFFLOW_MODEL_DIR` if set. On download a sidecar `<file>.json` records the URL, license, sha256 and date.

**Open issue:** pin sha256 hashes in `photo/models.py` before release. They are `None` now and could not be obtained offline.

## Benchmark data

Lid-driven cavity reference values are transcribed from Ghia, Ghia & Shin (1982), *J. Comput. Phys.* 48:387–411. They are published numerical results, used here only for comparison.

## Browser interface (Phase 5)

| Component | Version | License | Use |
|---|---|---|---|
| FastAPI | 0.162.0 | MIT | HTTP API framework (`reefflow/ui/server.py`) |
| Starlette | 1.10.0 | BSD-3-Clause | FastAPI's ASGI toolkit (static files, SSE streaming) |
| Pydantic | 2.14.2 | MIT | FastAPI dependency |
| pydantic-core | 2.50.2 | MIT | Pydantic dependency |
| uvicorn | 0.54.0 | BSD-3-Clause | Local ASGI server |
| h11 | 0.16.0 | MIT | uvicorn HTTP/1.1 implementation |
| click | 8.5.0 | BSD-3-Clause | uvicorn dependency |
| python-multipart | 0.0.32 | Apache-2.0 | File uploads (mesh / project import) |
| anyio | 4.15.1 | MIT | Starlette/FastAPI async runtime |
| typing-extensions, typing-inspection, annotated-types, annotated-doc | 4.16.0, 0.4.4, 0.8.0, 0.0.5 | PSF-2.0, MIT, MIT, MIT | FastAPI/Pydantic dependencies |
| colorama | 0.4.6 | BSD-3-Clause | click dependency on Windows |
| three.js | r160 (npm 0.160.0) | MIT | 3-D viewport. Vendored, unmodified, in `reefflow/ui/static/vendor/three/`: `three.module.js`, OrbitControls, TransformControls, STLLoader, PLYLoader. The LICENSE and SHA-256 hashes are in the same folder. |
| httpx | 0.28.1 | BSD-3-Clause | Tests only (FastAPI TestClient); not needed at runtime |
| httpcore, certifi, idna | 1.0.9, 2026.10.23, 3.20 | BSD-3-Clause, MPL-2.0, BSD-3-Clause | httpx dependencies (tests only) |

Notes
- certifi (MPL-2.0) is a test-only transitive dependency, and only its CA bundle file is used. MPL-2.0 permits commercial use; it requires source availability only for modified MPL files, and we do not modify it.
- FastAPI pulls in `opentelemetry-api` 1.45.2 (Apache-2.0) as an optional instrumentation hook; it does nothing unless configured.
- No fonts, icons or CSS frameworks are bundled. The UI uses system fonts, and its favicon is an inline SVG written for this project.

## Phase 4 data

The fixture library (`reefflow/light/data/fixture_library.yaml`) is original generic data; no manufacturer data were copied. Pure-water absorption values in `reefflow/light/config.py` are approximate literature values (Pope & Fry 1997; facts, cited in ASSUMPTIONS A-407). Phase 4 adds no dependencies.

## Network access

ReefFlow makes no network requests. Taichi's start-up update check (an HTTPS request) is disabled by the package (DECISIONS D-201).

The one exception is `reefflow.photo.models.download_model(...)`, which downloads an optional model's weights. It must be called explicitly with `accept_license=True`, refuses non-commercial models, and is disabled by `REEFFLOW_OFFLINE=1`.

## Equipment catalogue data (D-844)

`reefflow/data/pump_products.yaml`, `reefflow/data/overflow_products.yaml` and `reefflow/light/data/fixture_products.yaml` hold product specifications (flow, head, power, dimensions) and published PAR readings. These are facts, each listed with its source URL; no text, images, charts or files from manufacturers were copied into ReefFlow (chart values were read off and stored as numbers). Brand and model names are used only to identify the products (nominative use); ReefFlow is not affiliated with or endorsed by any of these companies. The research notes and build scripts are in `tools/catalog/`. No new dependencies.
