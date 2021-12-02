//**********************************************
//
//  ECG buffer management
//
//**********************************************

#ifndef BEAT_H_
#define BEAT_H_

#include <stdlib.h>
#include "data_type.h"
#include "feature_extract.h"

beat_t* new_beat(void);
int beat_set_delay(beat_t* beat, int16_t delay);
int beat_set_prev_beat(beat_t* beat, beat_t* prev_beat);
int beat_set_next_beat(beat_t* beat, beat_t* next_beat);
int beat_set_signal(beat_t* beat, int16_t* segment);
int beat_set_gold_label(beat_t* beat, int gold_label);
int16_t beat_get_delay(beat_t* beat);
beat_t* beat_get_prev_beat(beat_t* beat);
beat_t* beat_get_next_beat(beat_t* beat);
int16_t* beat_get_signal(beat_t* beat);
int beat_get_amplitude(beat_t* beat);
int beat_get_gold_label(beat_t* beat);
int16_t beat_inc_delay(beat_t* beat);
int delete_beat(beat_t* beat);
int print_beat(beat_t* beat);

#endif //BEAT_H_
