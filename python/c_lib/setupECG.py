from distutils.core import setup, Extension

module1 = Extension('ECGlib',
                    sources = ['c_src/detect_module.c','c_src/qrsdet.c','c_src/qrsfilt.c'],
                    include_dirs = ['c_src'])

setup (name = 'ECGlib',
       version = '1.0',
       description = 'ECG signal processing package',
       ext_modules = [module1])