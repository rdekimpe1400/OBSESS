//**********************************************
//
//  SVM header file
//
//**********************************************

#ifndef SVM_H_
#define SVM_H_

// Declaration
int svm_predict( int16_t* x);
 
extern const int n_feat_V;
extern const int n_sv_V;
extern const int n_feat_S;
extern const int n_sv_S;
extern const int feature_select_idx_V[];
extern const int feature_select_idx_S[];

#endif //SVM_H_
 