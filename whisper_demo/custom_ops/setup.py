from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_cuda_ops',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            'custom_cuda_ops',
            ['kernels.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
