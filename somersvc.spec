# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SomerSVC.

Build:  pyinstaller --noconfirm somersvc.spec
Output: dist/SomerSVC.app
"""
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_all,
)
import os

block_cipher = None

# Hidden imports — packages that PyInstaller can't auto-detect
hiddenimports = [
    'so_vits_svc_fork',
    'so_vits_svc_fork.inference',
    'so_vits_svc_fork.f0',
    'so_vits_svc_fork.preprocessing',
    'so_vits_svc_fork.train',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.tree._utils',
    'sklearn.utils._typedefs',
    'pyqtgraph',
    'librosa',
    'librosa.effects',
    'soundfile',
    'demucs',
    'demucs.separate',
    'demucs.pretrained',
    'paramiko',
    'runpod',
    'boto3',
    'requests',
    'PIL',
    'PIL.Image',
    'numpy',
    'scipy',
    'scipy.signal',
    'torch',
    'torchaudio',
    'transformers',
]

# Additional submodule collections — pulls everything they import dynamically
hiddenimports += collect_submodules('so_vits_svc_fork')
hiddenimports += collect_submodules('demucs')
hiddenimports += collect_submodules('librosa')
hiddenimports += collect_submodules('transformers', filter=lambda name: 'tf_' not in name)

# Data files to bundle (model weights, configs, fonts, icons, etc.)
datas = [
    ('assets', 'assets'),
]

# so-vits-svc-fork ships some package-data; collect it
try:
    datas += collect_data_files('so_vits_svc_fork')
except Exception:
    pass
try:
    datas += collect_data_files('librosa')
except Exception:
    pass
try:
    datas += collect_data_files('demucs')
except Exception:
    pass

# Packages with mypyc-compiled extensions or other surprise binaries — pull everything
binaries = []
for pkg in ('mypy', 'runpod', 'click', 'tomli', 'tomli_w'):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# mypyc compiles some packages to standalone top-level .so files at the
# site-packages root (named like <hash>__mypyc.cpython-NNN-darwin.so).
# These aren't inside a package, so collect_all/collect_data_files miss them.
import glob, sysconfig
site_pkgs = sysconfig.get_paths().get('purelib') or ''
if site_pkgs:
    for so in glob.glob(os.path.join(site_pkgs, '*__mypyc.cpython-*.so')):
        binaries.append((so, '.'))

# Things that bloat the binary needlessly — exclude
excludes = [
    'tkinter',
    'matplotlib.tests',
    'numpy.tests',
    'scipy.tests',
    'PIL.tests',
    'pandas.tests',
    'tensorflow',
    'tensorboard',
    'jax',
    'jaxlib',
]

a = Analysis(
    ['main.py'],
    pathex=[os.path.abspath(SPECPATH)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SomerSVC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # Use system arch (arm64 on Apple Silicon)
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.png',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SomerSVC',
)

app = BUNDLE(
    coll,
    name='SomerSVC.app',
    icon='assets/icon.png',
    bundle_identifier='com.somersaudio.somersvc',
    info_plist={
        'CFBundleName': 'SomerSVC',
        'CFBundleDisplayName': 'SomerSVC',
        'CFBundleShortVersionString': '1.0.14',
        'CFBundleVersion': '1.0.14',
        'NSHighResolutionCapable': 'True',
        'NSMicrophoneUsageDescription': 'SomerSVC needs microphone access for realtime voice conversion.',
        'NSAppleEventsUsageDescription': 'SomerSVC uses Apple Events for output folder access.',
        'LSMinimumSystemVersion': '12.0',
    },
)
