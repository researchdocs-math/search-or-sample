#!/usr/bin/env python3
"""Verify the SHA-256 checksum of each file in the released experiment package."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text())
failures=[]
for name,expected in manifest['sha256'].items():
 p=root/name
 if not p.is_file():failures.append(name+': absent');continue
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:failures.append(name+': checksum mismatch')
if failures:
 print('\n'.join(failures));sys.exit(1)
print(f"Verified {len(manifest['sha256'])} files.")
