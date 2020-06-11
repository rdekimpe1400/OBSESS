from distutils.core import setup, Extension

module1 = Extension('SVMintlib',
                    sources = ['c_src/svm_int_module.c','c_src/svm_int.c'],
                    include_dirs = ['c_src'])

setup (name = 'SVMintlib',
       version = '1.0',
       description = 'SVM classification',
       ext_modules = [module1])