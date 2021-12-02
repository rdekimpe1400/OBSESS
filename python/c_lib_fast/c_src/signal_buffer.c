/*****************************************************************************
FILE:  buffer.c
AUTHOR:	R. Dekimpe
REVISED:	06/2020

****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "signal_buffer.h"

//#define DEBUG_PRINT

// Global variables
int16_t* signalBuf;      // Signal buffer
int signalBufIdx;         // Signal buffer current position

int* labelBuf;      // Gold label buffer

// Initialization function
int init_signal_buffer(){
  // Init signal buffer
  signalBuf = (int16_t*)malloc((SIGNAL_BUFFER_LENGTH+SIGNAL_SEGMENT_LENGTH)*sizeof(int16_t));
  labelBuf = (int*)malloc((SIGNAL_BUFFER_LENGTH)*sizeof(int));
  if(signalBuf == NULL){
    return 0;
  }
  signalBufIdx = 0;
  
  return 0;
}

// Push new sample in the signal buffer
void push_sample(int16_t sample, int label_gold){ 
  // Store data
  signalBuf[signalBufIdx] = sample;
  if(signalBufIdx<SIGNAL_SEGMENT_LENGTH){
    signalBuf[signalBufIdx+SIGNAL_BUFFER_LENGTH] = sample;
  }
  labelBuf[signalBufIdx] = label_gold;
  // Update index
  signalBufIdx = signalBufIdx+1;
  if(signalBufIdx==SIGNAL_BUFFER_LENGTH){
    signalBufIdx = 0;
  }
}

// Get signal segment, from delay+SIGNAL_OUTPUT_BEFORE samples to delay-SIGNAL_OUTPUT_AFTER before current position to 
int16_t* get_segment(int delay){
  int startSeg;
  if(delay<SIGNAL_SEGMENT_AFTER){
    printf("Delay must be larger than window size\n");
    return NULL;
  }
  // Compute limit indices
  startSeg = signalBufIdx-(delay+SIGNAL_SEGMENT_BEFORE);
  if(startSeg<0){ // Loop back if before beginning of array
    startSeg = startSeg+SIGNAL_BUFFER_LENGTH;
  }
  return &(signalBuf[startSeg]);
}

// Get gold label at delay samples befores
int get_gold_label(int delay){
  int idx;
  // Compute limit indices
  idx = signalBufIdx-delay;
  if(idx<0){ // Loop back if before beginning of array
    idx = idx+SIGNAL_BUFFER_LENGTH;
  }
  return labelBuf[idx];
}

// Closing function
int close_signal_buffer(){
  free(signalBuf);
  free(labelBuf);
  return 1;
}
