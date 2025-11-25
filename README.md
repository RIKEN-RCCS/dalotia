# dalotia -- a data loader library for tensors in AI

![CTest CI Badge](https://github.com/RIKEN-RCCS/dalotia/actions/workflows/ctest.yml/badge.svg)
![Spack install CI Badge](https://github.com/RIKEN-RCCS/dalotia/actions/workflows/spack.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

A thin C++ / C / Fortran wrapper around whatever the next fancy tensor format is going to be.

- Simple installation
- Optimized loading (load zero-copy transpose, memory-mapped, ...)
- Currently supported formats: safetensors (planned: GGUF)
- Extensible in file and data formats

## Worked Example

Suppose you are training a fully connected neural network, defined with PyTorch like this:

```Python
class SingleLayerFullyConnectedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 300)
        self.fc2 = nn.Linear(300, 6)
        # [...initialization...]

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

After training, you could save the model weights by `safetensors.save_model(net, "model.safetensors")`.

Then, the inference could be transferred directly to C++, by using `dalotia` and a BLAS library like this:

```C++
std::string filename = "./model.safetensors";
auto dalotia_file = std::unique_ptr<dalotia::TensorFile>(
    dalotia::make_tensor_file(filename));
auto [weight_extents_1, weight_1] =
    dalotia_file->load_tensor_dense<float>("fc1.weight");
auto [bias_extents_1, bias_1] =
    dalotia_file->load_tensor_dense<float>("fc1.bias");
auto [weight_extents_1, weight_1] =
    dalotia_file->load_tensor_dense<float>("fc2.weight");
auto [bias_extents_2, bias_2] =
    dalotia_file->load_tensor_dense<float>("fc2.bias");
int num_hidden_neurons = weight_extents_1[0];
int num_input_features = weight_extents_1[1];
// [...initialize inputs,
//     allocate intermediate and output arrays...]

for (int i = 0; i < num_input_elements; ++i)
  for (int j = 0; j < num_hidden_neurons; ++j)
    hidden[i*num_hidden_neurons + j] = bias_1[j];

cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
  num_hidden_neurons, num_input_elements,
  num_input_features, 1.0, weight_1.data(),
  num_input_features, inputs.data(), num_input_features,
  1.0, hidden.data(), num_hidden_neurons);
for (auto& value : hidden)
  value = value < 0. ? 0. : value; // ReLU

// [...repeat GEMM for weight_2/bias_2...]
```

For Fortran, the same could be achieved like this:

```fortran
type(C_ptr) :: dalotia_file_pointer
! fixed-size input arrays
real(C_float) :: weight_1(10, 300), bias_1(300)
! allocatable input arrays
real(C_float), allocatable :: weight_2(:,:), bias_2(:)
integer :: num_input_features = size(weight_1, 1)
integer :: num_hidden_neurons = size(weight_1, 2)
! [...other variables...]

dalotia_file = dalotia_open_file("./model.safetensors")
call dalotia_load_tensor(dalotia_file, "fc1.bias", bias_1)
call dalotia_load_tensor(dalotia_file, &
                         "fc1.weight", weight_1)
call dalotia_load_tensor_dense(dalotia_file, &
                         "fc2.bias", bias_2)
call dalotia_load_tensor_dense(dalotia_file, &
                         "fc2.weight", weight_2)
call dalotia_close_file(dalotia_file)

! [...initialize inputs,
!     allocate intermediate and output arrays...]

do o = 1, num_input_elements
    fc1_output(:,o) = bias_1(:)
end do
call sgemm('T', 'N', num_hidden_neurons, &
  num_input_elements, num_input_features, 1.0, &
  weight_1, num_input_features, inputs, &
  num_input_features,  1.0, &
  fc1_output, num_hidden_neurons)
fc1_output = max(0.0, fc1_output) ! reLU

! [...repeat GEMM for weight_2/bias_2...]
```

### ...with shared-memory parallelism through OpenMP

Depending on tensor sizes, the efficiency of shared-memory programs on NUMA architectures can depend on
which part of the main memory the tensors live on.
The following code snippets are equivalent to the above but duplicates the weights and biases on each thread
and allocates slices of the input and output data for each thread.

```C++
// [...initialize inputs, allocate output arrays...]
std::string filename = "./model.st";
auto dalotia_file = std::unique_ptr<dalotia::TensorFile>(
    dalotia::make_tensor_file(filename));
#pragma omp parallel
{
// thread-local -> first-touch efficiency
auto [weight_extents_1, weight_1] = 
    dalotia_file->load_tensor_dense<float>("fc1.weight");
auto [bias_extents_1, bias_1] = 
    dalotia_file->load_tensor_dense<float>("fc1.bias");
auto [weight_extents_2, weight_2] = 
    dalotia_file->load_tensor_dense<float>("fc2.weight");
auto [bias_extents_2, bias_2] = 
    dalotia_file->load_tensor_dense<float>("fc2.bias");
int num_hidden_neurons = weight_extents_1[0];
int num_input_features = weight_extents_1[1];
// [...allocate thread-local intermediate arrays...]

for (int i = 0; i < this_thread_num_inputs; ++i)
  for (int j = 0; j < num_hidden_neurons; ++j)
    hidden[i*num_hidden_neurons + j] = bias_1[j];

cblas_sgemm(CblasColMajor, CblasTrans,
    CblasNoTrans, num_hidden_neurons, 
    this_thread_num_inputs, num_input_features, 
    1.0, weight_1.data(), num_input_features,
    &inputs[this_thread_inputs_start_index], 
    num_input_features,  1.0, hidden.data(),
    num_hidden_neurons);
for (auto& value : hidden)
  value = value < 0. ? 0. : value; // ReLU

// [...repeat GEMM for weight_2/bias_2...]
```

```fortran
type(C_ptr) :: dalotia_file_pointer
! fixed-size input arrays
real(C_float) :: weight_1(10, 300), bias_1(300)
! allocatable input arrays
real(C_float), allocatable :: weight_2(:,:), bias_2(:)
integer :: num_input_features = size(weight_1, 1)
integer :: num_hidden_neurons = size(weight_1, 2)
! [...other variables...]
! [...initialize inputs, allocate output arrays...]

dalotia_file = dalotia_open_file("./model.safetensors")
!$OMP parallel
call dalotia_load_tensor(dalotia_file, &
                         "fc1.bias", bias_1)
call dalotia_load_tensor(dalotia_file, &
                         "fc1.weight", weight_1)
call dalotia_load_tensor_dense(dalotia_file, & 
                         "fc2.bias", bias_2)
call dalotia_load_tensor_dense(dalotia_file, &
                         "fc2.weight", weight_2)
! [...allocate thread-local intermediate arrays...]

do o = 1, this_thread_num_inputs
    fc1_output(:,o) = bias_1(:)
end do
call sgemm('T', 'N', num_hidden_neurons, &
    this_thread_num_inputs, num_input_features, &
    1.0,  weight_1, num_input_features, inputs, &
    num_input_features, 1.0, fc1_output, num_hidden_neurons)        
fc1_output = max(0.0, fc1_output) ! reLU

! [...repeat GEMM for weight_2/bias_2...]
!$OMP end parallel
call dalotia_close_file(dalotia_file)
```

This is exactly what's used in the fully-connected [C++](https://github.com/RIKEN-RCCS/dalotia_evaluation/blob/main/benchmarks/SubgridLES/subgridLES.cpp)
and [Fortran examples](https://github.com/RIKEN-RCCS/dalotia_evaluation/blob/main/benchmarks/SubgridLES/subgridLES.f90)
of the inference comparison benchmark code https://github.com/RIKEN-RCCS/dalotia_evaluation.

## Installation

### With CMake

Requires: CMake >= 3.24

```bash
git clone https://github.com/RIKEN-RCCS/dalotia.git
cd dalotia
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/../install ..
make
make install
```

(or adapt the paths accordingly)

Then, use it in your CMake project with `find_package(dalotia)`.

Additional CMake options are

- `DALOTIA_CPP_BUILD_EXAMPLES`, default ON
- `DALOTIA_BUILD_TESTS`, default ON
- `DALOTIA_WITH_CPP_PMR`, default ON
- `DALOTIA_WITH_OPENMP`, default OFF
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
git clone https://github.com/RIKEN-RCCS/dalotia.git
cd dalotia
spack repo add $(pwd)/spack_repo_dalotia # registers this folder for finding package info
spack spec dalotia # to see the dependency tree
spack info dalotia # to see a description of all variants
spack install dalotia # to install dalotia and all dependencies
```

If you want to make local modifications to `dalotia`, this here may be more convenient:

```bash
git clone https://github.com/RIKEN-RCCS/dalotia.git
cd dalotia
spack repo add $(pwd)/spack_repo_dalotia
spack dev-build dalotia@main
```

(you need to re-run the last line every time you want to install an updated version.)

Find more details on customizing builds in the
[Spack documentation](https://spack.readthedocs.io/en/latest/repositories.html).
