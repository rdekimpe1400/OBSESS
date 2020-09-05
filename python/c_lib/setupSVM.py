from distutils.core import setup, Extension

module1 = Extension('SVMlib',
                    sources = ['c_src/svm_module.c','c_src/svm.c'],
                    include_dirs = ['c_src'])

setup (name = 'SVMlib',
       version = '1.0',
       description = 'SVM classification',
       ext_modules = [module1])