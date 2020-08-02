//**********************************************
//
//  ECG digital processing
//
//**********************************************

#ifndef ECG_H_
#define ECG_H_

#include <stdint.h>

int16_t* ECG_wrapper( int sample, int* delay, int* output);
void ECG_init();
void ECG_close();

#endif //ECG_H_
