//**********************************************
//
//  DWT decomposition
//
//**********************************************

#ifndef DWT_FILTER_H_
#define DWT_FILTER_H_

#include <stdint.h>
// DWT model data

// Wavelet type : db4
const int F_l = 8;
const int F_h = 8;
const int16_t filter_l[8] = {-174,539,505,-3064,-458,10336,11712,3775};
const int16_t filter_h[8] = {-3775,11712,-10336,-458,3064,505,-539,-174};
int shift_dwt = 14;


#endif //DWT_FILTER_H_
