"""Create an allowlisted, self-contained source bundle, without user outputs.

Produces dist/scgvb-12pi.tar.gz plus SHA256. No binaries/wheels are bundled;
install platform-specific dependencies on Linux aarch64 at the destination.
"""
import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def validate_local_imports(content):
    """Reject missing local modules, relative imports and package initializers."""
    def check_module(module, importer):
        parts = module.split('.')
        for index in range(1, len(parts) + 1):
            base = '/'.join(parts[:index])
            for dependency in (base + '.py', base + '/__init__.py'):
                if (ROOT / dependency).is_file() and dependency not in content:
                    raise ValueError(f'Bundle omits {dependency}, imported by {importer}')

    for name, data in content.items():
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse(data, filename=name)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = name.split('/')[:-1]
                    prefix = package[:len(package) - node.level + 1]
                    module = '.'.join(prefix + ([node.module] if node.module else []))
                else:
                    module = node.module or ''
                modules = [module] + [module + '.' + alias.name for alias in node.names
                                      if alias.name != '*']
            else:
                continue
            for module in modules:
                check_module(module, name)


def build(output):
    names = [
        'calc_molecule.py','calc_diatomic_mol.py','generate_orbitals.py','build_hpc_bundle.py',
        'pyproject.toml','scgvb/io/orbital_viewer.html',
        'assets/ayatori-logo.png',
        'examples/legacy_generate_cube.py',
        'README.md','INPUT_FILE.md','ACTIVE_SPACE.md','HPC_README.md','MOLECULE_INTERFACE.md','GEOMETRY_OPTIMIZATION.md',
        'JAX_BACKEND.md','ORBITAL_VIEWER.md','PHYSICS_FIX.md',
        'SECOND_ORDER.md','CORE_RESPONSE.md','OPTIMIZATION_STATE_RETENTION.md',
        'requirements-base.txt','requirements-cpu.txt','requirements-gpu-cuda13.txt','requirements-geometry.txt',
    ]
    for pattern in ('scgvb/**/*.py','scgvb/**/*.md','tools/*.py','hpc/*.py','hpc/*.sh',
                    'examples/*.xyz','examples/inputs/*.toml'):
        names.extend(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern)
                     if not (p.parent == ROOT/'hpc' and p.name.lower().startswith('rikyu')))
    # This local input depends on a saved rikyu job, rather than shipped examples.
    names = sorted(set(names) - {'examples/inputs/b2h6_geometry_bobyqa.toml'})
    content = {name: (ROOT/name).read_bytes() for name in names}
    validate_local_imports(content)
    manifest={'format':1,'description':'SCGVB single-GPU 12-pi-electron source distribution',
              'sha256':{name:hashlib.sha256(data).hexdigest() for name,data in content.items()}}
    content['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name,data in sorted(content.items()):
                info=tarfile.TarInfo('scgvb-12pi/'+name)
                info.size=len(data)
                info.mode=0o755 if name.endswith('.sh') else 0o644
                archive.addfile(info,io.BytesIO(data))
    checksum=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name+'.sha256').write_text(f'{checksum}  {output.name}\n')
    print(f'{output.resolve()} ({output.stat().st_size} bytes, {len(content)} files)')
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/scgvb-12pi.tar.gz')
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
