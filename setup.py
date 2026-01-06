from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
import platform
import shutil

class FlatbuffersExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_flatbuffers(self):

        install_dir = os.path.abspath(os.path.join(self.build_temp, 'flatbuffers-install'))
        try:
            subprocess.check_output([os.path.join(install_dir, 'bin', 'flatc'), '--version'])
            return None  
        except OSError:
            print("Building Flatbuffers from source...")
            code_url = 'https://github.com/google/flatbuffers/archive/refs/tags/v24.3.25.zip'
            code_dir = os.path.abspath(os.path.join(self.build_temp, 'flatbuffers-source'))
            build_dir = os.path.abspath(os.path.join(self.build_temp, 'flatbuffers-build'))
            
            # Clean any previous failed builds
            if os.path.exists(code_dir):
                shutil.rmtree(code_dir)
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir)
            if os.path.exists(install_dir):
                shutil.rmtree(install_dir)
            
            os.makedirs(code_dir, exist_ok=True)
            os.makedirs(build_dir, exist_ok=True)
            os.makedirs(install_dir, exist_ok=True)
            
            # Download and verify the zip file exists
            zip_path = os.path.join(code_dir, 'flatbuffers.zip')
            subprocess.check_call(['curl', '-L', code_url, '-o', zip_path])
            
            if not os.path.exists(zip_path):
                raise RuntimeError(f"Failed to download Flatbuffers source to {zip_path}")
            
            # Extract with better error handling
            try:
                subprocess.check_call(['unzip', '-o', '-q', zip_path, '-d', code_dir])
            except subprocess.CalledProcessError as e:
                print(f"Failed to extract {zip_path} to {code_dir}")
                raise
            
            # Build flatbuffers with install prefix
            cmake_args = [
                'cmake',
                os.path.join(code_dir, 'flatbuffers-24.3.25'),
                f'-DCMAKE_INSTALL_PREFIX={install_dir}',
                '-G', 'Unix Makefiles',
                '-DCMAKE_BUILD_TYPE=Release'
            ]
            
            try:
                subprocess.check_call(cmake_args, cwd=build_dir)
                subprocess.check_call(['cmake', '--build', '.'], cwd=build_dir)
                subprocess.check_call(['cmake', '--install', '.'], cwd=build_dir)
            except subprocess.CalledProcessError as e:
                print(f"Failed to build Flatbuffers in {build_dir}")
                raise
            
            return install_dir

    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed")
        
        # Build or verify flatbuffers installation
        flatbuffers_dir = self.build_flatbuffers()
        
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # Get the build directory for temporary files
        build_temp = os.path.abspath(self.build_temp)
        
        # Get the installation directory (site-packages)
        # This is where we want our .so files to end up
        install_dir = os.path.dirname(os.path.abspath(
            self.get_ext_fullpath(ext.name)
        ))
        
        if not hasattr(self, '_flatbuffers_dir'):
            self._flatbuffers_dir = self.build_flatbuffers()
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_temp}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]
        
        if self._flatbuffers_dir:
            cmake_args.extend([
                f'-DFLATBUFFERS_ROOT={self._flatbuffers_dir}',
                f'-DFLATC_EXECUTABLE={os.path.join(self._flatbuffers_dir, "bin", "flatc")}'
            ])
            
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # Unix/Linux only configuration
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        build_args += ['--', '-j2']
        
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)
        
        # Determine library extension based on platform
        if sys.platform == 'darwin':
            # Main module is built as MODULE (.so bundle), helper libs as SHARED (.dylib)
            main_ext = '.so'  # CMake MODULE produces .so on macOS
            helper_ext = '.dylib'
            target_ext = '.so'
        elif sys.platform == 'win32':
            main_ext = '.pyd'
            helper_ext = '.pyd'
            target_ext = '.pyd'
        else:
            main_ext = '.so'
            helper_ext = '.so'
            target_ext = '.so'

        package_dir = os.path.join(install_dir, 'sauerkraut')
        os.makedirs(package_dir, exist_ok=True)

        # Copy main module
        main_src = os.path.join(build_temp, f'_sauerkraut{main_ext}')
        if os.path.exists(main_src):
            dst = os.path.join(package_dir, f'_sauerkraut{target_ext}')
            self.copy_file(main_src, dst)

        # Copy helper libraries
        for lib_name in ['greenlet_compat', 'serdes']:
            src = os.path.join(build_temp, f'{lib_name}{helper_ext}')
            if os.path.exists(src):
                target_name = f'{lib_name}{target_ext}'
                dst = os.path.join(package_dir, target_name)
                self.copy_file(src, dst)

    def _get_python_library(self):
        """Helper function to find the Python library path."""
        # Check common library name patterns
        library_names = [
            f'libpython{sys.version_info.major}.{sys.version_info.minor}.so',
            f'libpython{sys.version_info.major}.{sys.version_info.minor}d.so',  # Debug version
            f'libpython{sys.version_info.major}.{sys.version_info.minor}m.so',  # With malloc debugging
            f'python{sys.version_info.major}.{sys.version_info.minor}.dll',  # Windows
        ]
        
        lib_dirs = [
            os.path.join(sys.prefix, 'lib'),
            os.path.join(sys.prefix, 'libs'),
            os.path.join(sys.base_prefix, 'lib'),
            os.path.join(sys.base_prefix, 'libs'),
        ]
        
        # Try to find the library
        for lib_dir in lib_dirs:
            for name in library_names:
                library_path = os.path.join(lib_dir, name)
                if os.path.exists(library_path):
                    return library_path
        
        return None

    def get_outputs(self):
        """Get the path to the built library."""
        # Get the built library path from the CMake build directory
        if sys.platform == "win32":
            lib_name = "_sauerkraut.pyd"
        else:
            lib_name = "_sauerkraut.so"
        
        # Look in the CMake build directory
        library_path = os.path.join(self.build_temp, lib_name)
        
        # Also check lib.* directory as a fallback
        if not os.path.exists(library_path):
            # Find any lib.* directory
            build_dir = os.path.dirname(self.build_temp)
            lib_dirs = [d for d in os.listdir(build_dir) if d.startswith('lib.')]
            if lib_dirs:
                alt_path = os.path.join(build_dir, lib_dirs[0], lib_name)
                if os.path.exists(alt_path):
                    library_path = alt_path
        
        if not os.path.exists(library_path):
            raise RuntimeError(f"Built library not found at {library_path}")
        
        return [library_path]

setup(
    name='sauerkraut',
    version='0.1',
    author='Zane Fink',
    author_email='zanef2@illinois.edu',
    description='Serialization for Python\'s Control State',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=[CMakeExtension('_sauerkraut')],
    cmdclass={
        'build_ext': CMakeBuild,
    },
    zip_safe=False,
    python_requires='>=3.13',
    install_requires=[
        'bytecode',
        'numpy'
    ]
) 