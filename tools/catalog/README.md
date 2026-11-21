# Equipment catalogue sources (D-844)

`research/*.yaml` are the raw research notes (2026-11-12): every value as published, with
source URLs and confidence notes. The build scripts turn them into the catalogue files the
app loads:

    cd tools/catalog
    ../../.venv/Scripts/python build_overflows.py   # -> reefflow/data/overflow_products.yaml
    ../../.venv/Scripts/python build_powerheads.py  # -> research/_powerheads_built.yaml
    ../../.venv/Scripts/python build_returns.py     # -> reefflow/data/pump_products.yaml (powerheads + returns)
    ../../.venv/Scripts/python build_lights.py      # -> reefflow/light/data/fixture_products.yaml

`fit_par.py` fits a fixture's photon output (and beam) to published PAR readings with the
light tracer's emitter model. To add a product, add it to the research file, add any
hand-read numbers the build script asks for, rebuild, and run `pytest tests/test_catalog.py`.
