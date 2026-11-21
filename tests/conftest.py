import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def backend_name():
    from reefflow import backend
    return backend.init(os.environ.get("REEFFLOW_ARCH", "auto"))


@pytest.fixture(autouse=True)
def _collect_taichi_garbage_on_main_thread():
    """Free leftover objects after every test, on the main thread.

    Solver tests can leave Taichi ndarrays in reference cycles. If Python's cyclic
    garbage collector happens to run inside a TestClient worker thread (a later UI
    test), Taichi frees the ndarray from that thread and aborts the whole process
    ("Fatal Python error: Aborted" in taichi/_ndarray.py __del__). The UI server is
    not affected (simulation jobs run in their own processes)."""
    yield
    import gc
    gc.collect()
