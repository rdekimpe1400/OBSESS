
#include <stdio.h>
#include <Python.h>

double* wavedec(double* input, int inlen, int level, int* outlen);

PyObject *Convert_Big_Array(double* array, int length)
  { PyObject *pylist, *item;
    int i;
    pylist = PyList_New(length);
    for (i=0; i<length; i++) {
      item = PyFloat_FromDouble(array[i]);
      PyList_SetItem(pylist, i, item);
    }
    return pylist;
  }

static PyObject* multi_dwt(PyObject* self, PyObject* args)
{
    PyObject* seq;
    int seqlen;
    int i;
    double* x;
    double* out;
    int outlen = 0;
    
    if (!PyArg_ParseTuple(args, "O", &seq))
      return NULL;
    seq = PySequence_Fast(seq, "argument must be iterable");
    if(!seq)
        return NULL;
    /* prepare data as an array of doubles */
    seqlen = PySequence_Fast_GET_SIZE(seq);
    //if(seqlen!=n_feat){
    //  printf("Wrong number of features, detected %d but require %d\n",seqlen, n_feat);
    //  return NULL;
    //}
    x = malloc(seqlen*sizeof(double));
    if(!x) {
        Py_DECREF(seq);
        return PyErr_NoMemory(  );
    }
    for(i=0; i < seqlen; i++) {
        PyObject *fitem;
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        if(!item) {
            Py_DECREF(seq);
            free(x);
            return 0;
        }
        fitem = PyNumber_Float(item);
        if(!fitem) {
            Py_DECREF(seq);
            free(x);
            PyErr_SetString(PyExc_TypeError, "all items must be numbers");
            return 0;
        }
        x[i] = PyFloat_AS_DOUBLE(fitem);
        Py_DECREF(fitem);
    }    

    /* clean up, compute, and return result */
    Py_DECREF(seq);
    out = wavedec(x, seqlen, 4, &outlen);
    free(x);
    
    PyObject* out_o = Convert_Big_Array(out, outlen);
    
    //return Py_BuildValue("i", 42);
    return out_o;
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    { "multi_dwt", multi_dwt, METH_VARARGS, "Multi-level wavelet transform" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef DWTlib = {
    PyModuleDef_HEAD_INIT,
    "DWTlib",
    "DWT transform",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_DWTlib(void)
{
    return PyModule_Create(&DWTlib);
}
