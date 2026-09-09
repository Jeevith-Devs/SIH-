# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Build Specification for ASTRA-Track standalone Windows Executable.
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

added_files = [
    ('configs', 'configs'),
    ('docs', 'docs'),
    ('README.md', '.'),
    ('requirements.txt', '.')
]

hidden_imports = [
    'cv2',
    'numpy',
    'scipy',
    'scipy.spatial',
    'sklearn',
    'sklearn.utils._typedefs',
    'yaml',
    'fpdf',
    'matplotlib',
    'matplotlib.backends.backend_tkagg',
    'PIL',
    'PIL.ImageTk',
    'astra_track'
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ASTRA-Track',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
