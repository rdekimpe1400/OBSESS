//**********************************************
//
//  SVM header file
//
//**********************************************

#ifndef SVM_H_
#define SVM_H_

// Declaration
int svm_predict( int16_t* x, int32_t* decision );

extern const int n_feat;
extern const int n_sv;
extern const int feature_select_idx[];

#endif //SVM_H_
 