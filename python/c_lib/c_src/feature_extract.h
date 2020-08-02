//**********************************************
//
//  Features extraction
//
//**********************************************

#ifndef FEATURES_H_
#define FEATURES_H_

#include <stdint.h>

// Parameter definition
#define FEATURES_COUNT_RR   4
#define FEATURES_COUNT_TIME 25
#define FEATURES_DWT_START  98
#define FEATURES_DWT_END    133
#define FEATURES_COUNT_DWT  (FEATURES_DWT_END-FEATURES_DWT_START)
#define N_CHANNELS          1
#define FEATURES_COUNT      FEATURES_COUNT_RR+N_CHANNELS*(FEATURES_COUNT_TIME+FEATURES_COUNT_DWT)

#define FEATURES_SELECT_COUNT   12

#define DWT_LEVEL 4
#define DWT_BEFORE 60
#define DWT_AFTER 60
#define DWT_LENGTH DWT_BEFORE+DWT_AFTER

// Function declaration
int16_t* extract_features(void);
int16_t* select_features(int16_t* feature_all);

#endif //FEATURES_H_
