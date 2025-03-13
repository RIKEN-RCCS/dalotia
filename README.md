# dalotia -- a data loader library for tensors in AI

![CTest CI Badge](https://github.com/RIKEN-RCCS/dalotia/actions/workflows/ctest.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

A thin C++ / C / Fortran wrapper around whatever the next fancy tensor format is going to be.

- Simple installation
- Optimized loading (load zero-copy transpose, memory-mapped, ...)
- Currently supported formats: safetensors (planned: GGUF)
- Extensible in file and data formats

## Installation

### With CMake

Requires: CMake >= 3.24

Run this in the cloned dalotia folder (or adapt the paths accordingly):

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PATH=$(pwd)/../install ..
make
make install
```

Then, use it in your CMake project with `find_package(dalotia)`.

Additional CMake options are

- `DALOTIA_CPP_BUILD_EXAMPLES`, default ON
- `DALOTIA_BUILD_TESTS`, default ON
- `DALOTIA_WITH_CPP_PMR`, default ON
- `DALOTIA_WITH_OPENMP`, default ON
- `DALOTIA_WITH_SAFETENSORS_CPP`, default ON
- `DALOTIA_WITH_FORTRAN`, default ON

so for example, to disable building the Fortran interface, you would call `cmake` as

```bash
cmake -DDALOTIA_WITH_FORTRAN=OFF ..
```

### With Spack

dalotia can also be installed through the [Spack HPC package manager](https://github.com/spack/spack/).
Assuming you have configured spack on your system, and the shell integration is activated (e.g. through
a [script](https://spack.readthedocs.io/en/latest/packaging_guide.html#interactive-shell-support)),
you can run the following

```bash
spack repo add $(pwd)/spack_repo_dalotia # registers this folder for finding package info
spack spec dalotia # to see the dependency tree
spack info dalotia # to see a description of all variants
spack install dalotia # to install dalotia and all dependencies
```

Find more details on customizing builds in the 
[Spack documentation](https://spack.readthedocs.io/en/latest/repositories.html).

