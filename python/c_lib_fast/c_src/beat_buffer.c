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
#include "beat_buffer.h"
#include "feature_extract.h"

//#define DEBUG_PRINT

// Global variables
beat_t** beatBuf;  // Delay (in samples) from the unclassificed beats
int beatBufCnt;           // Number of unclassified beats in buffer


// Initialization function
int init_beat_buffer(){
  int i;
  // Init beat buffer
  beatBuf = (beat_t**)malloc(BEAT_BUFFER_LENGTH*sizeof(beat_t*));
  if(beatBuf == NULL){
    return 0;
  }
  for(i=0; i<BEAT_BUFFER_LENGTH; i++){
    beatBuf[i] = NULL;
  }
  beatBufCnt = 0;
  
  return 0;
}

// Push new beat in the beat delay buffer
void add_beat(int16_t delay){ 
#ifdef DEBUG_PRINT
  printf("Add beat\n");
#endif
  if(beatBufCnt==BEAT_BUFFER_LENGTH){
    printf("Buffer is full\n");
    return;
  }
  
  // Store beat
  beatBuf[beatBufCnt] = new_beat();
  beat_set_delay(beatBuf[beatBufCnt], delay);
  
  // Link beats
  if(beatBufCnt>0){
    beat_set_prev_beat(beatBuf[beatBufCnt], beatBuf[beatBufCnt-1]);
    beat_set_next_beat(beatBuf[beatBufCnt-1], beatBuf[beatBufCnt]);
  }
  
  // Update index
  beatBufCnt++;
}

// Increment delays in the beat delay buffer by 1
void increment_beat_delay(){ 
  int i = 0;
  for(i=0;i<beatBufCnt;i++){
    beat_inc_delay(beatBuf[i]);
  }
}

// Check if beat is ready for processing
int is_beat_ready(){ 
  if(beatBufCnt<2){ // Next beat needs to be detected
    return 0;
  }
  if(beatBuf[0]->delay<SIGNAL_SEGMENT_AFTER){ // Sufficient time window for feature extraction
    return 0;
  }
#ifdef DEBUG_PRINT
  printf("Beat ready\n");
#endif
  return 1;
}

// Pop oldest beat
uint16_t pop_beat(int* gold_label){ 
#ifdef DEBUG_PRINT
  printf("Pop beat\n");
#endif
  int i = 0;
  uint16_t delay = beatBuf[0]->delay;
  *gold_label = beatBuf[0]->gold_label;
  delete_beat(beatBuf[0]);
  for(i=1; i<beatBufCnt; i++){
    beatBuf[i-1] = beatBuf[i];
  }
  beatBufCnt = beatBufCnt-1;
  return delay;
}

// Closing function
int close_beat_buffer(){ 
  free(beatBuf);
  return 1;
}

// Extract features from oldest beat
int16_t* buffer_get_features(int* amplitude){ 
int16_t* features;
#ifdef DEBUG_PRINT
  printf("Get beat\n");
#endif
  beat_set_signal(beatBuf[0], get_segment(beatBuf[0]->delay));
  *amplitude = beat_get_amplitude(beatBuf[0]);
  beat_set_gold_label(beatBuf[0], get_gold_label(beatBuf[0]->delay));
  extract_features_RR(beatBuf[0]);
  extract_features_time(beatBuf[0]);
  extract_features_DWT(beatBuf[0]);
  print_beat(beatBuf[0]);
  features = get_features(beatBuf[0]);
  
  update_feature_template(beatBuf[0]);
  
  return features;
}
