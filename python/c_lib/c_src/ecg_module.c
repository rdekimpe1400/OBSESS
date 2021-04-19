
#include <Python.h>
#include <stdlib.h>
#include <stdint.h>
#include "ecg.h" 
#include "feature_extract.h" 
#include "svm.h" 

extern const int n_sv;
extern const int n_feat;

PyObject *Create_Output_Dict(int16_t* features, int length, int delay, int class)
  { 
    PyObject *features_list, *item, *output;
    int i;
    features_list = PyList_New(length);
    for (i=0; i<length; i++) {
      item = PyLong_FromLong((long)features[i]);
      PyList_SetItem(features_list, i, item);
    }
    output = PyDict_New();
    PyDict_SetItemString(output, "features", features_list);
    PyDict_SetItemString(output, "delay", PyLong_FromLong(delay));
    PyDict_SetItemString(output, "class", PyLong_FromLong(class));
    return output;
  }
  
PyObject *Create_Config_Dict(int svm_n_sv, int svm_n_feat)
  { 
    PyObject *output;
    output = PyDict_New();
    PyDict_SetItemString(output, "n_sv", PyLong_FromLong(svm_n_sv));
    PyDict_SetItemString(output, "n_feat", PyLong_FromLong(svm_n_feat));
    return output;
  }
  
// ECG sample processing function 
static PyObject* process_sample(PyObject* self, PyObject* args)
{
    int sample;
    int delay;
    int output;
    
    if (!PyArg_ParseTuple(args, "i", &sample))
      return NULL;
    
    ECG_wrapper(sample,0,&delay,&output);
    
    return PyLong_FromLong(output);
}

// ECG sample processing with feature extraction
static PyObject* compute_features(PyObject* self, PyObject* args)
{
    int sample;
    int label_gold;
    int delay;
    int output;
    int16_t* features;
    
    if (!PyArg_ParseTuple(args, "ii", &sample,&label_gold))
      return NULL;
    features=ECG_wrapper(sample,label_gold,&delay,&output); 
     
    PyObject* out_o;
    if(output>0){
      out_o = Create_Output_Dict(features, FEATURES_COUNT, delay, output);
    }else{
      out_o = Py_None;
      Py_INCREF(Py_None);
    }
    
    return out_o;
}

// Initialization function
static PyObject* init(PyObject* self, PyObject* args)
{
    ECG_init();
    
    PyObject* out_o;
    out_o = Create_Config_Dict(n_sv,n_feat);
    
    return out_o;
}

// Closing function
static PyObject* finish(PyObject* self, PyObject* args)
{
    ECG_close();
    Py_INCREF(Py_None);
    return Py_None;
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    { "process_sample", process_sample, METH_VARARGS, "Processes a new data sample (detection, feature extraction and classification)" },
    { "compute_features", compute_features, METH_VARARGS, "Processes a new data sample and returns features" },
    { "init", init, METH_NOARGS, "Initializes the processing function" },
    { "finish", finish, METH_NOARGS, "Closes the processing function" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef ECGlib = {
    PyModuleDef_HEAD_INIT,
    "ECGlib",
    "ECG signal processing",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_ECGlib(void)
{
    return PyModule_Create(&ECGlib);
}
