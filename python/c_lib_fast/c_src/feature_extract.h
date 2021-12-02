//**********************************************
//
//  Features extraction
//
//**********************************************

#ifndef FEATURES_H_
#define FEATURES_H_

#include <stdint.h>
#include "data_type.h"
#include "beat.h"
#include "signal_buffer.h"

// Parameter definition
#define FEATURES_COUNT_RR   8
#define FEATURES_COUNT_TIME 30
#define FEATURES_DWT_START  0
#define FEATURES_DWT_END    146
#define FEATURES_COUNT_DWT  (FEATURES_DWT_END-FEATURES_DWT_START)
#define N_CHANNELS          1
#define FEATURES_COUNT      (FEATURES_COUNT_RR+N_CHANNELS*(FEATURES_COUNT_TIME+FEATURES_COUNT_DWT))

#define FEATURES_SMOOTH_COUNT   8
#define FEATURES_SMOOTH_COUNT_LOG   3

#define FEATURES_SMOOTH_DECAY_LOG   4

#define DWT_LEVEL 4
#define DWT_BEFORE 60
#define DWT_AFTER 60
#define DWT_LENGTH DWT_BEFORE+DWT_AFTER

#define DEFAULT_RR              0.9*SAMPLING_FREQUENCY

#define FEATURES_RR_RATIO_SHIFT 8


// Function declaration
features_t* new_features(void);
int16_t* get_features(beat_t* beat);
int extract_features_RR(beat_t* beat);
int extract_features_time(beat_t* beat);
int extract_features_DWT(beat_t* beat);
int gather_features(beat_t* beat, int16_t* features_out);
void smooth_features_init(int length);
int16_t* deviation_features(int16_t* input_features, int length);
int16_t* get_features_buffer(void);
int init_features_buffer(void);
int delete_features(features_t* features);
int update_feature_template(beat_t* beat);
int close_features_buffer(void);

#endif //FEATURES_H_

