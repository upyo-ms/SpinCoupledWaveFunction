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

ROOT = Path(__file__).resolve().parent

def validate_local_imports(content):
    """Reject a distribution that omits an imported top-level project module."""
    for name, data in content.items():
        if not name.endswith('.py'):
            continue
        tree = ast.parse(data, filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                dependency = module.split('.')[0] + '.py'
                if (ROOT / dependency).is_file() and dependency not in content:
                    raise ValueError(f'Bundle omits {dependency}, imported by {name}')

def build(output):
    names = [
        'RestartOrbitals.py','ActiveSpace.py','AOSpace.py','PopulationAnalysis.py','CalculationInput.py','CoreOptimization.py','PiSpace.py','CompoundMinors.py','DeterminantEnergy.py',
        'GetMoleculeInfor.py','InitialOrbitals.py','InitialSpin.py','OptimizationDiagnostics.py','AlternatingOptimization.py','WavefunctionOptimization.py','CoupledOptimization.py','SpinResponseCorrection.py','SingleOrbitalOptimization.py','NonorthgonalEnergy.py',
        'NonorthgonalEnergyJAX.py','StableEnergyJAX.py','calc_molecule.py',
        'calc_diatomic_mol.py','generate_orbitals.py','orbital_viewer.html',
        'benchmark_evaluators.py','benchmark_jax.py','benchmark_active_space.py','BenchmarkProblems.py','build_hpc_bundle.py',
        'README.md','INPUT_FILE.md','ACTIVE_SPACE.md','HPC_README.md','MOLECULE_INTERFACE.md',
        'JAX_BACKEND.md','ORBITAL_VIEWER.md','PHYSICS_FIX.md',
        'requirements-base.txt','requirements-cpu.txt','requirements-gpu-cuda13.txt',
    ]
    for pattern in ('Spin_Function/*.py','Spin_Function/*.md','hpc/*.py','hpc/*.sh',
                    'examples/*.xyz','examples/inputs/*.toml'):
        names.extend(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern))
    names=sorted(set(names))
    content={name:(ROOT/name).read_bytes() for name in names}
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

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/scgvb-12pi.tar.gz')
    build(parser.parse_args().output)
