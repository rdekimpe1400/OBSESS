from distutils.core import setup, Extension

module1 = Extension('QRSlib',
                    sources = ['c_src/qrs_module.c','c_src/qrsdet.c','c_src/qrsfilt.c'],
                    include_dirs = ['c_src'])

setup (name = 'QRSlib',
       version = '1.0',
       description = 'QRS detection package',
       ext_modules = [module1])