//**********************************************
//
//  ECG digital processing
//
//**********************************************

#ifndef ECG_H_
#define ECG_H_

#include <stdint.h>

int16_t* ECG_wrapper( int sample, int label_gold, int* delay, int* output, int* gold_label);
void ECG_init();
void ECG_close();

#endif //ECG_H_
