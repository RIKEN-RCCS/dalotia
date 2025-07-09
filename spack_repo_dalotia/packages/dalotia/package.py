# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class Dalotia(CMakePackage):
    """Data loader for tensors in AI library """

    homepage = "https://github.com/RIKEN-RCCS/dalotia"
    git = "https://github.com/RIKEN-RCCS/dalotia"

    maintainers("freifrauvonbleifrei", "domke")

    license("Apache-2.0", checked_by="freifrauvonbleifrei")

    # FIXME: Add proper versions here.
    version("main", branch="main")
    version("1.0.0", tag="v1.0.0")

    variant("cpp_pmr", default=True, description="use polymorphic memory resources (pmr) C++17 feature for dalotia")
    variant("openmp", default=True, description="Build with OpenMP support")
    variant("safetensorscpp", default=True, description="use safetensors-cpp for tensor I/O")
    variant("fortran", default=True, description="Build Fortran interface")

    depends_on("cxx", type="build")
    depends_on("c", type="build")
    depends_on("fortran", type="build", when="+fortran")
    depends_on("cmake@3.24:", type="build")
    depends_on("safetensors-cpp+cxxexceptions", when="+safetensorscpp")


    def cmake_args(self):
        args = [
            self.define("DALOTIA_CPP_BUILD_EXAMPLES", True),
            self.define_from_variant("DALOTIA_WITH_CPP_PMR", "cpp_pmr"),
            self.define_from_variant("DALOTIA_WITH_OPENMP", "openmp"),
            self.define_from_variant("DALOTIA_WITH_SAFETENSORS_CPP", "safetensorscpp"),
            self.define_from_variant("DALOTIA_WITH_FORTRAN", "fortran"),
        ]
        if self.spec.satisfies("+safetensorscpp"):
            args.append(self.define("safetensors-cpp_DIR", "find"))
        return args

    def setup_dependent_build_environment(self, env, dependent_spec):
        cmake_prefix = self.prefix.lib.cmake.dalotia
        env.set('dalotia_DIR', cmake_prefix)

    def setup_run_environment(self, env):
        return self.setup_dependent_build_environment(env, None)
