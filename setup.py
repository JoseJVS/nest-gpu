from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import os
from pathlib import Path

class cmake_build(build_ext):
    def run(self):
        # The path where CMake will be configured and Arbor will be built.
        build_directory = os.path.abspath(self.build_temp)
        # The path where the package will be copied after building.
        lib_directory = os.path.abspath(self.build_lib)
        print("WHAT ARE THE DIRS?", build_directory, lib_directory, os.getcwd())
        #for i in dir(self):
        #    meth = getattr(self, i)
        #    try:
        #        print(i,meth())
        #    except:
        #        print(i,meth)
        print(self.get_source_files())
        # Where to copy the package after it is built, so that whatever the next phase is
        # can copy it into the target 'prefix' path.
        dest_path = lib_directory + '/nestgpu'
        print(os.path.abspath('nestgpu/libnestgpukernel.so'), dest_path)

        # Copy from build path to some other place from whence it will later be installed.
        # ... or something like that
        # ... setuptools is an enigma monkey patched on a mystery
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)
            
        print("is_file", Path(os.getcwd() + "/src/libnestgpukernel.so").is_file())
        self.copy_file('src/libnestgpukernel.so', dest_path)

        #print("is_file", Path(os.getcwd() + "/pythonlib/libnestgpukernel.so").is_file())
setup(
    name="nestgpu",
    version="0.1",
    packages=["nestgpu"],
    ext_modules=[
        Extension(
            name="libnestgpukernel",  # as it would be imported
                               # may include packages/namespaces separated by `.`

            sources=[], # all sources are compiled into a single binary file
        ),
    ],
    cmdclass={
        'build_ext': cmake_build,
    }
)
