from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='softmax_metal',
    ext_modules=[
        CppExtension(
            name='softmax_metal_backend',
            sources=['softmax_all_in_one.mm'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [],
                'cpp': ['-stdlib=libc++', '-framework', 'Metal', '-framework', 'Foundation']
            },
            extra_link_args=['-framework', 'Metal', '-framework', 'Foundation']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)