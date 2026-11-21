"""ReefFlow: lattice Boltzmann CFD and photometric analysis for reef aquariums."""
import os as _os

# Taichi starts a background HTTPS "check for updates" request on ti.init().
# ReefFlow must run fully offline, and on this workstation's TLS-inspecting
# network that thread aborts the process (OPENSSL_Applink). Disable it before
# any module imports Taichi (DECISIONS D-201).
_os.environ.setdefault("TI_SKIP_VERSION_CHECK", "ON")
