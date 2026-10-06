"""Run on an allocated compute node, inside the intended Python environment."""
import argparse
import importlib.metadata
import json
import platform
import os
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('gpu','cpu'), default='gpu')
    args=parser.parse_args()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    devices=jax.devices(args.device)
    device=devices[0]
    # Place explicitly; default-device configuration cannot redirect this probe.
    x=jax.device_put(jnp.arange(16,dtype=jnp.float64).reshape(4,4),device)
    value,gradient=jax.jit(jax.value_and_grad(lambda z: jnp.sum((z@z)**2)))(x)
    jax.block_until_ready((value,gradient))
    if gradient.dtype != jnp.float64 or not bool(jnp.isfinite(gradient).all()):
        raise RuntimeError('FP64 autodiff probe failed')
    print(json.dumps({'machine':platform.machine(),'python':platform.python_version(),
                      'requested_backend':args.device,'selected_device':str(device),
                      'device_kind':device.device_kind,'visible_devices':[str(d) for d in devices],
                      'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'versions':{n:importlib.metadata.version(n) for n in ('numpy','scipy','pyscf','jax','jaxlib')},
                      'float64_autodiff':'passed','memory_stats':device.memory_stats()},indent=2))

if __name__=='__main__':
    try:
        main()
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f'Environment check failed (no fallback): {exc}',file=sys.stderr)
        raise SystemExit(1)
