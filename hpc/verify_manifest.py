"""Verify the exact distributed source bytes; run from any working directory."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST.json').read_text())
for name,expected in manifest['sha256'].items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch: {name}')
print(f"Verified {len(manifest['sha256'])} files")
