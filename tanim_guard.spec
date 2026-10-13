# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Sipat.exe (onedir). Built by build_exe.ps1.

The app runs the .tflite model with LiteRT, so TensorFlow/Keras are excluded -
they are only needed to TRAIN (scripts/). Uploads are not bundled: the exe
writes them to data/ next to itself (desktop.py).
"""

from PyInstaller.utils.hooks import collect_all

litert_datas, litert_binaries, litert_hidden = collect_all('ai_edge_litert')

a = Analysis(
    ['desktop.py'],
    pathex=['.'],
    binaries=litert_binaries,
    datas=[
        ('app/templates', 'app/templates'),
        ('app/static/css', 'app/static/css'),
        ('app/static/js', 'app/static/js'),
        ('app/ml/model/tanim_guard.tflite', 'app/ml/model'),
        ('app/ml/model/labels.json', 'app/ml/model'),
        ('app/ml/model/cam_weights.npy', 'app/ml/model'),
        ('app/ml/model/evaluation_report.json', 'app/ml/model'),
    ] + litert_datas,
    hiddenimports=litert_hidden,
    excludes=['tensorflow', 'keras', 'matplotlib', 'tkinter', 'pytest', 'IPython'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Sipat',
          console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name='Sipat', upx=False)
