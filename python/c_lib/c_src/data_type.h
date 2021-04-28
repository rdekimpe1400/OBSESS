//**********************************************
//
//  ECG buffer management
//
//**********************************************

#ifndef DATA_H_
#define DATA_H_

#include <stdlib.h>


typedef struct features_t
{
	int16_t prev2_RR;
	int16_t prev_RR;
	int16_t next_RR;
	int16_t avrg_RR;
	int16_t prev_diff_RR;
	int16_t next_diff_RR;
	int16_t ratio1_RR;
	int16_t ratio2_RR;
  
  int time_len;
  int16_t* time;
  
  int dwt_len;
  int16_t* dwt;
  int16_t* dwt_diff;
  
} features_t;

typedef struct beat_t
{
	int16_t delay;
  
	struct beat_t* prev_beat;
	struct beat_t* next_beat;
  
  int16_t* signal;
  
  int gold_label;
  
	features_t* features;
  
} beat_t;

#endif //DATA_H_
