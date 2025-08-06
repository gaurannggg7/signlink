# -*- mode: python -*-
block_cipher = None

a = Analysis([
    'utils.py',
    'whisper_transcribe.py',
    'gemma_loader.py',
    'gloss_builder.py',
    'mapping.py',
    'renderer.py',
    'demo_cli.py',
    'app.py'
],
    pathex=[],
    binaries=[
        ('models/gemma3n_E2B', 'models/gemma3n_E2B'),
        ('content/asl_app_data/dictionary', 'content/asl_app_data/dictionary'),
        ('content/asl_app_data/filtered_unique_asl_words.csv', 'content/asl_app_data')
    ],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name='asl_app', console=True)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas,
               strip=False, upx=True, name='asl_app')
