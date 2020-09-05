from distutils.core import setup, Extension

module1 = Extension('DWTlib',
                    sources = ['c_src/dwt_module.c','c_src/dwt.c'],
                    include_dirs = ['c_src'])

setup (name = 'DWTlib',
       version = '1.0',
       description = 'DWT transform',
       ext_modules = [module1])