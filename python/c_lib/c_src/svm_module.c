
#include <stdio.h>
#include <Python.h>

int svm_predict( double* x, double* decision );

extern int n_feat;

static PyObject* predict(PyObject* self, PyObject* args)
{
    PyObject* seq;
    int seqlen;
    int i;
    double decision[3];
    double* x;
    int label;
    
    if (!PyArg_ParseTuple(args, "O", &seq))
      return NULL;
    seq = PySequence_Fast(seq, "argument must be iterable");
    if(!seq)
        return NULL;
    /* prepare data as an array of doubles */
    seqlen = PySequence_Fast_GET_SIZE(seq);
    if(seqlen!=n_feat){
      printf("Wrong number of features, detected %d but require %d\n",seqlen, n_feat);
      return NULL;
    }
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
    label = svm_predict(x, decision);
    free(x);
    
    //return Py_BuildValue("[ddd]", decision[0], decision[1], decision[2]);
    return Py_BuildValue("i", label);
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    { "predict", predict, METH_VARARGS, "Predicts class of a new sample using SVM model" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef SVMlib = {
    PyModuleDef_HEAD_INIT,
    "SVMlib",
    "SVM classification",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_SVMlib(void)
{
    return PyModule_Create(&SVMlib);
}
