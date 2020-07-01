
#include <Python.h>
#include <stdlib.h>
#include <stdint.h>
#include "ecg.h" 
#include "feature_extract.h" 

PyObject *Create_Output_Dict(int16_t* features, int length, int delay)
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
    return output;
  }
  
// ECG sample processing function 
static PyObject* process_sample(PyObject* self, PyObject* args)
{
    int sample;
    int output;
    
    if (!PyArg_ParseTuple(args, "i", &sample))
      return NULL;
    
    ECG_wrapper(sample,&output);
    
    return PyLong_FromLong(output);
}

// ECG sample processing with feature extraction
static PyObject* compute_features(PyObject* self, PyObject* args)
{
    int sample;
    int output;
    int16_t* features;
    
    if (!PyArg_ParseTuple(args, "i", &sample))
      return NULL;
    
    features=ECG_wrapper(sample,&output);
    
    PyObject* out_o;
    if(output>0){
      out_o = Create_Output_Dict(features, FEATURES_COUNT, output);
    }else{
      out_o = Py_None;
    }
    
    return out_o;
}

// Initialization function
static PyObject* init(PyObject* self, PyObject* args)
{
    ECG_init();
    return Py_None;
}

// Closing function
static PyObject* finish(PyObject* self, PyObject* args)
{
    ECG_close();
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
