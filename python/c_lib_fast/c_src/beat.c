/*****************************************************************************
FILE:  buffer.c
AUTHOR:	R. Dekimpe
REVISED:	06/2020

****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "signal_buffer.h"
#include "beat.h"
#include "feature_extract.h" 

//#define DEBUG_PRINT

// Global variables

// Opening function
beat_t* new_beat(){
  beat_t* beat = (beat_t*) malloc(sizeof(beat_t));
  
  beat->delay = 0;
  
  beat->prev_beat = NULL;
  beat->next_beat = NULL;
  
  beat->signal = (int16_t*) malloc(SIGNAL_SEGMENT_LENGTH*sizeof(int16_t));
  
  beat->features = new_features();
  
	return beat;
}

// Set local variables
int beat_set_delay(beat_t* beat, int16_t delay){
  beat->delay = delay;
	return 0;
}

int beat_set_prev_beat(beat_t* beat, beat_t* prev_beat){
  beat->prev_beat = prev_beat;
	return 0;
}

int beat_set_next_beat(beat_t* beat, beat_t* next_beat){
  beat->next_beat = next_beat;
	return 0;
}

int beat_set_signal(beat_t* beat, int16_t* segment){
  memcpy( beat->signal, segment, SIGNAL_SEGMENT_LENGTH*sizeof(int16_t));
	return 0;
}

int beat_set_gold_label(beat_t* beat, int gold_label){
  beat->gold_label = gold_label;
	return 0;
}

// Get local variables
int16_t beat_get_delay(beat_t* beat){
	return beat->delay;
}

beat_t* beat_get_prev_beat(beat_t* beat){
	return beat->prev_beat;
}

beat_t* beat_get_next_beat(beat_t* beat){
	return beat->next_beat;
}

int16_t* beat_get_signal(beat_t* beat){
	return beat->signal;
}

int beat_get_gold_label(beat_t* beat){
	return beat->gold_label;
}

int beat_get_amplitude(beat_t* beat){
  int i=0;
  int min=0, max=0;
  for(i=SIGNAL_SEGMENT_BEFORE-30;i<SIGNAL_SEGMENT_BEFORE+30;i++){
    if (beat->signal[i]<max){
      min = beat->signal[i];
    }
    if (beat->signal[i]>max){
      max = beat->signal[i];
    }
  }
	return max-min;
}

// Increment delay
int16_t beat_inc_delay(beat_t* beat, int incr){
  beat->delay += incr;
	return 0;
}

// Closing function
int delete_beat(beat_t* beat){
  if(beat->next_beat!=NULL){
    beat->next_beat->prev_beat = NULL;
  }
  if(beat->prev_beat!=NULL){
    beat->prev_beat->next_beat = NULL;
  }
  delete_features(beat->features);
  free(beat->signal);
  free(beat);
  return 0;
}

// Print beat
int print_beat(beat_t* beat){
#ifdef DEBUG_PRINT
  printf("Beat - Delay %d\n", beat->delay);
  printf("       Signal [%d %d %d ...]\n", beat->signal[0], beat->signal[1], beat->signal[2]);
   
#endif
	return 0;
}
