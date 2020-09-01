
#include <Python.h>
#include <stdint.h>
#include "qrsdet.h"

// QRS detection function
static PyObject* detect(PyObject* self, PyObject* args)
{
    int ecg;
    int ecg_filt;
    int detection;
    
    if (!PyArg_ParseTuple(args, "ii", &ecg,&ecg_filt))
      return NULL;
    
    detection=QRSDet(ecg, ecg_filt,0 );
    
    return PyLong_FromLong(detection);
}

// ECG transformation function for QRS detection
static PyObject* filter(PyObject* self, PyObject* args)
{
    int data;
    uint32_t dummy;
    
    if (!PyArg_ParseTuple(args, "i", &data))
      return NULL;
    
    data=QRSFilter(data, 0 , &dummy);
    
    return PyLong_FromLong(data);
}

// Initialization of filtering functions
static PyObject* init(PyObject* self, PyObject* args)
{
    uint32_t dummy;
    QRSDet( 0, 0, 1 );
    QRSFilter( 0, 1,&dummy );
    return Py_None;
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    { "detect", detect, METH_VARARGS, "Detects QRS peak in ECG signal" },
    { "filter", filter, METH_VARARGS, "Transforms ECG signal for QRS detection" },
    { "init", init, METH_NOARGS, "Initializes the transformation functions" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef QRSlib = {
    PyModuleDef_HEAD_INIT,
    "QRSlib",
    "QRS detection",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_QRSlib(void)
{
    return PyModule_Create(&QRSlib);
}
