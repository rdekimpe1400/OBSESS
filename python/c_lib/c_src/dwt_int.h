//**********************************************
//
//  DWT decomposition
//
//**********************************************

#ifndef DWT_H_
#define DWT_H_

#include <stdint.h>
// DWT model data

#define MAX_LEVELS  5

int16_t* wavedec(int16_t* input, int inlen, int level, int* outlen);
int dwt_bufferinit(int N, int level);
int* dwt_bufferlength(void);

#endif //DWT_H_
