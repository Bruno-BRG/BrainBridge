# -*- mode: python ; coding: utf-8 -*-
"""
BrainBridge – PyInstaller spec file
Gera um executável Windows one-folder em dist/BrainBridge/
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(SPECPATH).resolve()
PACKAGE = REPO_ROOT / "brainbridge_v2"
RESOURCES = PACKAGE / "resources"

# ---------------------------------------------------------------------------
# Hidden imports – heavy scientific stack needs explicit listing
# ---------------------------------------------------------------------------
hidden = []
hidden += collect_submodules("numpy")
hidden += collect_submodules("scipy")
hidden += collect_submodules("sklearn")
hidden += collect_submodules("pandas")
hidden += collect_submodules("matplotlib")
hidden += collect_submodules("seaborn")
hidden += collect_submodules("pyzmq")
hidden += collect_submodules("serial")       # pyserial
hidden += collect_submodules("dateutil")     # python-dateutil

# TensorFlow (optional – comment out the next 2 lines to skip)
try:
    hidden += collect_submodules("tensorflow")
except Exception:
    print("⚠ TensorFlow not found – skipping")

# Our own package
hidden += collect_submodules("brainbridge_v2")

# ---------------------------------------------------------------------------
# Data files – resources bundled alongside the exe
# ---------------------------------------------------------------------------
datas = []

# Resource files (HTML, etc.)
if RESOURCES.exists():
    datas.append((str(RESOURCES), "brainbridge_v2/resources"))

# Matplotlib needs its data directory
datas += collect_data_files("matplotlib")
datas += collect_data_files("seaborn")

# ---------------------------------------------------------------------------
# Dynamic libraries
# ---------------------------------------------------------------------------
binaries = []
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("scipy")

try:
    binaries += collect_dynamic_libs("tensorflow")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(PACKAGE / "__main__.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "_tkinter",
        "xmlrunner",
        "IPython",
        "notebook",
        "jupyterlab",
    ],
    noarchive=False,
)

# ---------------------------------------------------------------------------
# Remove duplicate / unnecessary files to save space
# ---------------------------------------------------------------------------
# (PyInstaller sometimes pulls in duplicate .pyd / .dll)
seen = set()
deduped = []
for item in a.binaries:
    key = item[0].lower()
    if key not in seen:
        seen.add(key)
        deduped.append(item)
a.binaries = deduped

# ---------------------------------------------------------------------------
# PYZ + EXE + COLLECT (one-folder mode)
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BrainBridge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # GUI application – no console window
    icon=None,              # TODO: set path to .ico when available
    version_info=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BrainBridge",
)
