# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack.package import *


class SafetensorsCpp(CMakePackage):
    """Header-only safetensors loader and saver in C++ """

    homepage = "https://github.com/syoyo/safetensors-cpp"
    git = "https://github.com/freifrauvonbleifrei/safetensors-cpp" #TODO: change to upstream

    # FIXME: Add a list of GitHub accounts to
    # notify when the package is updated.
    # maintainers("syoyo", "freifrauvonbleifrei")
    maintainers("freifrauvonbleifrei")

    license("MIT", checked_by="freifrauvonbleifrei")

    version("main", branch="add_cmake_install") #TODO

    variant("c", default=True, description="build C API")
    variant("examples", default=True, description="build examples")
    variant("cxxexceptions", default=False, description="enable C++ exceptions")

    depends_on("cxx", type="build")
    depends_on("c", type="build", when="+c")
    depends_on("cmake@3.16:", type="build")

    def cmake_args(self):
        args = [
            self.define_from_variant("SAFETENSORS_CPP_BUILD_C_API", "c"),
            self.define_from_variant("SAFETENSORS_CPP_BUILD_EXAMPLES", "examples"),
            self.define_from_variant("SAFETENSORS_CPP_CXX_EXCEPTIONS", "cxxexceptions"),
        ]
        return args
