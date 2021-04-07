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
	int16_t prev_RR;
	int16_t next_RR;
	int16_t avrg_RR;
  
  int16_t* time;
  
} features_t;

typedef struct beat_t
{
	int16_t delay;
  
	struct beat_t* prev_beat;
	struct beat_t* next_beat;
  
  int16_t* signal;
  
	features_t* features;
  
} beat_t;

#endif //DATA_H_
