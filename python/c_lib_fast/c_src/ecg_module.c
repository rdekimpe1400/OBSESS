
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "ecg.h" 
#include "feature_extract.h" 
#include "svm.h" 

extern const int n_sv_V;
extern const int n_feat_V;
extern const int n_sv_S;
extern const int n_feat_S;

int sample_buffer_length;
int16_t* sample_buffer;
int16_t* label_gold_buffer;

#define N_OUTPUT_MAX 10000
int* det_time;
int* det_label;
int* det_label_gold;

FILE *fp_feat;
int save_features;
int subset;

PyObject *Create_Output_Dict(int* det_time_arr, int* det_label_arr, int* det_label_gold_arr, int n_det)
  {     
    PyObject *det_time_list, *det_label_list, *det_label_gold_list, *item, *output;
    int i;
    det_time_list = PyList_New(n_det);
    det_label_list = PyList_New(n_det);
    det_label_gold_list = PyList_New(n_det);
    for (i=0; i<n_det; i++) {
      item = PyLong_FromLong((long)det_time_arr[i]);
      PyList_SetItem(det_time_list, i, item);
      item = PyLong_FromLong((long)det_label_arr[i]);
      PyList_SetItem(det_label_list, i, item);
      item = PyLong_FromLong((long)det_label_gold_arr[i]);
      PyList_SetItem(det_label_gold_list, i, item);
    }
    output = PyDict_New();
    PyDict_SetItemString(output, "det_time", det_time_list);
    PyDict_SetItemString(output, "det_label", det_label_list);
    PyDict_SetItemString(output, "det_label_gold", det_label_gold_list);
    return output;
  }
  
PyObject *Create_Config_Dict(int svm_n_sv_S, int svm_n_feat_S, int svm_n_sv_V, int svm_n_feat_V)
  { 
    PyObject *output;
    output = PyDict_New();
    PyDict_SetItemString(output, "n_sv_V", PyLong_FromLong(svm_n_sv_V));
    PyDict_SetItemString(output, "n_feat_V", PyLong_FromLong(svm_n_feat_V));
    PyDict_SetItemString(output, "n_sv_S", PyLong_FromLong(svm_n_sv_S));
    PyDict_SetItemString(output, "n_feat_S", PyLong_FromLong(svm_n_feat_S));
    return output;
  }
  
// ECG sample processing function 
static PyObject* process_sample(PyObject* self, PyObject* args)
{
    int sample;
    int delay;
    int output;
    int gold_label;
    
    if (!PyArg_ParseTuple(args, "i", &sample))
      return NULL;
    
    ECG_wrapper(sample,0,&delay,&output,&gold_label);
    
    return PyLong_FromLong(output);
}

// ECG sample processing with feature extraction
static PyObject* compute_features(PyObject* self, PyObject* args)
{
    int delay;
    int output;
    int gold_label;
    int16_t* features;
    int i=0, j=0;
    int det_idx = 0;
    PyObject *in_o;
    PyObject *in_o2;
    
    PyArg_ParseTuple(args, "OO", &in_o, &in_o2);
    PyObject *iter = PyObject_GetIter(in_o);
    PyObject *iter2 = PyObject_GetIter(in_o2);
    
    i=0;
    while (1) {
      PyObject *next = PyIter_Next(iter);
      if (!next) {
        break;
      }
      sample_buffer[i] = (int16_t) PyLong_AsLong(next);
      i++;
    }
    i=0;
    while (1) {
      PyObject *next = PyIter_Next(iter2);
      if (!next) {
        break;
      }
      label_gold_buffer[i] = (int16_t) PyLong_AsLong(next);
      i++;
    }
    
    for(i=0; i<sample_buffer_length; i++){
      //printf("%d\n",sample_buffer[i]);
      //printf("%d\n",label_gold_buffer[i]);
      features=ECG_wrapper(sample_buffer[i],label_gold_buffer[i],&delay,&output,&gold_label); 
      if(delay>0){
        det_time[det_idx] = i-delay;
        det_label[det_idx] = output;
        det_label_gold[det_idx] = gold_label;
        det_idx++;
        if(save_features){
          fprintf(fp_feat, "%d,%d,%d", subset, i-delay, gold_label);
          for(j=0; j<FEATURES_COUNT; j++){
            fprintf(fp_feat, ",%d", features[j]);
          }
          fprintf(fp_feat, "\n");
        }
      }
    }    
     
    PyObject* out_o;
    out_o = Create_Output_Dict(det_time, det_label, det_label_gold, det_idx);
    
    
    return out_o;
}

// Initialization function
static PyObject* init(PyObject* self, PyObject* args)
{
    char* feature_file_name;
    if (!PyArg_ParseTuple(args, "iiis", &sample_buffer_length, &save_features, &subset, &feature_file_name)){
      return NULL;
    }
    
    if(save_features){
      fp_feat = fopen(feature_file_name, "a");
    }
   
    sample_buffer = (int16_t*) malloc(sample_buffer_length*sizeof(int16_t));
    label_gold_buffer = (int16_t*) malloc(sample_buffer_length*sizeof(int16_t));
    
    det_time = (int*) malloc(N_OUTPUT_MAX*sizeof(int));
    det_label = (int*) malloc(N_OUTPUT_MAX*sizeof(int));
    det_label_gold = (int*) malloc(N_OUTPUT_MAX*sizeof(int));
    
    ECG_init();
    
    PyObject* out_o;
    out_o = Create_Config_Dict(n_sv_S,n_feat_S,n_sv_V,n_feat_V);
    
    return out_o;
}

// Closing function
static PyObject* finish(PyObject* self, PyObject* args)
{
    if(save_features){
      fclose(fp_feat);
    }
    free(sample_buffer);
    free(label_gold_buffer);
    free(det_time);
    free(det_label);
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
    { "init", init, METH_VARARGS, "Initializes the processing function" },
    { "finish", finish, METH_NOARGS, "Closes the processing function" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef ECGlib_fast = {
    PyModuleDef_HEAD_INIT,
    "ECGlib_fast",
    "ECG signal processing",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_ECGlib_fast(void)
{
    return PyModule_Create(&ECGlib_fast);
}
