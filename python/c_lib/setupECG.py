from distutils.core import setup, Extension

module1 = Extension('ECGlib',
                    sources = ['c_src/ecg_module.c','c_src/ecg.c','c_src/qrsdet.c','c_src/qrsfilt.c','c_src/buffer.c','c_src/dwt_int.c','c_src/feature_extract.c'],
                    include_dirs = ['c_src'])

setup (name = 'ECGlib',
       version = '1.0',
       description = 'ECG processing package',
       ext_modules = [module1])