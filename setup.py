# This file is part of crosci, licensed under the Academic Public License.
# See LICENSE.txt for more details.

import os
import platform

from Cython.Build import cythonize
from setuptools import Extension, setup

dir_path = os.path.dirname(os.path.realpath(__file__))

os_name = platform.system()
processor_name = platform.processor()

LIBOMP_PATH_ARM = "/opt/homebrew/opt/libomp"
LIBOMP_PATH_X86 = "/usr/local/opt/libomp"
LIBOMP_PATH_GITHUB = "/usr/local"

if os_name == "Windows":
    extra_compile_args = ["/openmp"]
    extra_link_args = []
elif os_name == "Darwin":
    if os.environ.get("LIBOMP_GITHUB") == "1":
        extra_compile_args = [
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{LIBOMP_PATH_GITHUB}/include",
        ]
        extra_link_args = [f"-L{LIBOMP_PATH_GITHUB}/lib", "-lomp"]
    elif processor_name == "arm":
        extra_compile_args = [
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{LIBOMP_PATH_ARM}/include",
        ]
        extra_link_args = [f"-L{LIBOMP_PATH_ARM}/lib", "-lomp"]
    else:
        extra_compile_args = [
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{LIBOMP_PATH_X86}/include",
        ]
        extra_link_args = [f"-L{LIBOMP_PATH_X86}/lib", "-lomp"]
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

extensions = [
    Extension(
        "crosci.run_DFA",
        [
            "src/crosci/run_DFA.pyx",
            "src/crosci/dfa.c",
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "crosci.run_fEI",
        [
            "src/crosci/run_fEI.pyx",
            "src/crosci/fEI.c",
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=cythonize(extensions),
)
