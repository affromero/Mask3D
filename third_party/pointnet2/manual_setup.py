# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os.path as osp

this_dir = osp.dirname(osp.abspath(__file__))

_ext_src_root = "pointnet2/_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob("{}/src/*.cu".format(_ext_src_root))
# _ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

def run_command(*args):
    subprocess.check_call(args)

run_command("pip", "uninstall", "pointnet2", "-y")

setup(
    name="pointnet2",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": [
                    "-O2",
                    "-I{}".format("{}/include".format(_ext_src_root)),
                ],
                "nvcc": [
                    "-O2",
                    "-I{}".format("{}/include".format(_ext_src_root)),
                ],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
)
