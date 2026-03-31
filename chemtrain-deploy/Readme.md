# Connector

The connector compiles a JAX model to StableHLO/HLO from Python and exposes a
shared-library interface that can be used from MD engines such as LAMMPS.

## Supported Stack

This checkout is now intended for the modernized deployment stack used by
`env_cueq_allegro_opt`:

- `jax==0.9.1`
- `jaxlib==0.9.1`
- `cuequivariance-jax==0.9.0`

The supported deployment flow is to build the connector and the GPU PJRT plugin
from the same checkout. Copying a wheel-provided PJRT plugin into `lib/` is
deprecated and should be used only as a debugging fallback.

## Build Connector

Create and activate an environment that matches the training/export stack. On
JUWELS Booster this is the existing `env_cueq_allegro_opt` environment.

Build the connector with:

```bash
python build.py
```

## Build GPU PJRT Plugin

### Generic build

Use a source build so the connector and PJRT plugin come from the same XLA/JAX
checkout:

```bash
python build.py --build_gpu_pjrt_plugin --enable_cuda --cuda_version <cuda-version>
```

### JUWELS Booster build

The supported JUWELS path is the dedicated build profile, which applies the
cluster defaults for CUDA 12.6 and cuDNN 9.5 and keeps CUDA compilation on the
stable NVCC path:

```bash
python build.py --build_profile juwels-booster --build_gpu_pjrt_plugin --enable_cuda
```

If you need to inspect the resolved options, add `--verbose`.

### Deprecated fallback

A wheel-provided PJRT plugin can still be copied in through the helper command:

```bash
python build.py --load_gpu_pjrt_plugin
```

That path is deprecated for the modernized stack because it can introduce PJRT
API mismatches between the connector framework and the plugin.

## Build LAMMPS Plugin

From the connector directory create and enter a build directory and compile the
LAMMPS plugin:

```bash
mkdir -p build
cd build
cmake -D LAMMPS_HEADER_DIR=<path/to/lammps/src> ../lammps_plugin
cmake --build .
```

Rebuild from scratch after connector-side changes with:

```bash
cmake --build . --clean-first
```

## Build LAMMPS With Plugin Support

Configure or rebuild LAMMPS with plugin support enabled:

```bash
cmake -D PKG_PLUGIN=yes ../cmake
cmake --build . -j <number_of_cores>
```

## Runtime Environment

At runtime you need:

- `PATH` pointing at the chosen LAMMPS build
- `LAMMPS_PLUGIN_PATH` pointing at the connector `build/` directory
- `JCN_PJRT_PATH` pointing at a directory that contains exactly one matching
  PJRT plugin

A minimal activation script looks like this:

```bash
#! /bin/bash

export PATH=<path/to/lammps/build>:$PATH
export LAMMPS_PLUGIN_PATH=<path/to/chemtrain-deploy/build>
export JCN_PJRT_PATH=<path/to/chemtrain-deploy/lib/pjrt_single>
```

The helper script in this repository creates `pjrt_single/` and exposes only
one plugin there so the connector cannot accidentally load duplicate or stale
PJRT shared libraries.

## Model Runtime Expectations

Models exported from the `env_cueq_allegro_opt` stack are expected to load
without submit-time MLIR rewriting. If an export compatibility issue appears,
fix it on the export side or in the connector compatibility layer rather than
mutating the protobuf in the LAMMPS submit script.
