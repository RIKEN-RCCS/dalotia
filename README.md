# dalotia -- a data loader library for tensors in AI

[![CTest CI Badge](https://github.com/RIKEN-RCCS/dalotia/actions/workflows/ctest.yml/badge.svg)]

## Features

A thin C++ / C / Fortran wrapper around whatever the next fancy tensor format is going to be.

- Simple installation
- Optimized loading (load zero-copy transpose, memory-mapped, ...)
- Currently supported formats: safetensors (planned: GGUF)
- Extensible in file and data formats

## Installation

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

## CMake options

<!-- !TODO -->
