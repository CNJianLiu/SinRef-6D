import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import glob

_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))
cuda_include_dir = os.path.join(CUDA_HOME, "include") if CUDA_HOME is not None else None
include_dirs = [os.path.join(_ext_src_root, "include")]
if cuda_include_dir is not None:
    include_dirs.append(cuda_include_dir)

setup(
    name='pointnet2',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-O3",
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                    "-I{}".format(os.path.join(_ext_src_root, "include")),
                ] + (["-I{}".format(cuda_include_dir)] if cuda_include_dir is not None else [])
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)
