/*****************************************************************************
FILE:  buffer.c
AUTHOR:	R. Dekimpe
REVISED:	06/2020

****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "buffer.h"

// Global variables
int16_t* signalBuf;      // Signal buffer
int signalBufIdx;         // Signal buffer current position
int16_t signalOutBuf[SIGNAL_OUTPUT_LENGTH];   // Output signal buffer

uint16_t* beatBufDelay;  // Delay (in samples) from the unclassificed beats
int beatBufCnt;           // Number of unclassified beats in buffer

uint16_t* RRBuf;         // Buffer of RR intervals of the previous beats

// Initialization function
int init_buffers(){
  int i = 0;
  //printf("Start init\n");
  // Init signal buffer
  signalBuf = (int16_t*)malloc(SIGNAL_BUFFER_LENGTH*sizeof(int16_t));
  if(signalBuf == NULL){
    return 0;
  }
  signalBufIdx = 0;
  
  // Init beat buffer
  beatBufDelay = (uint16_t*)malloc(BEAT_BUFFER_LENGTH*sizeof(uint16_t));
  if(beatBufDelay == NULL){
    return 0;
  }
  for(i = 0; i<BEAT_BUFFER_LENGTH; i++){
    beatBufDelay[i] = 0;
  }
  beatBufCnt = 0;
  
  // Init RR buffer
  RRBuf = (uint16_t*)malloc(RR_BUFFER_LENGTH*sizeof(uint16_t));
  if(RRBuf == NULL){
    return 0;
  }
  for(i = 0; i<RR_BUFFER_LENGTH; i++){
    RRBuf[i] = DEFAULT_RR;
  }
  
  return 1;
}

// Push new sample in the signal buffer
void push_sample(int16_t sample){ 
  // Store data
  signalBuf[signalBufIdx] = sample;
  // Update index
  signalBufIdx = signalBufIdx+1;
  if(signalBufIdx==SIGNAL_BUFFER_LENGTH){
    signalBufIdx = 0;
  }
}

// Get signal segment, from delay+SIGNAL_OUTPUT_BEFORE samples to delay-SIGNAL_OUTPUT_AFTER before current position to 
int16_t* get_segment(int delay){
  int start;
  int end;
  int i = 0;
  
  if(delay<SIGNAL_OUTPUT_AFTER){
    printf("Delay must be larger than window size\n");
    return NULL;
  }
  
  // Compute limit indices
  start = signalBufIdx-(delay+SIGNAL_OUTPUT_BEFORE);
  if(start<0){
    start = start+SIGNAL_BUFFER_LENGTH;
  }
  end = signalBufIdx-(delay-SIGNAL_OUTPUT_AFTER);
  if(end<0){
    end = end+SIGNAL_BUFFER_LENGTH;
  }
  // Copy data
  if(start<end){
    for(i = 0; i<SIGNAL_OUTPUT_LENGTH; i++){
      signalOutBuf[i] = signalBuf[i+start];
    }
  }else{
    for(i = 0; i<(SIGNAL_BUFFER_LENGTH-start); i++){
      signalOutBuf[i] = signalBuf[i+start];
    }
    for(i = 0; i<end; i++){
      signalOutBuf[i] = signalBuf[i+SIGNAL_BUFFER_LENGTH-start];
    }
  }
  
  return signalOutBuf;
}

// Push new beat in the beat delay buffer
void add_beat(int16_t delay){ 

  if(beatBufCnt==BEAT_BUFFER_LENGTH){
    printf("Buffer is full\n");
    return;
  }
  
  if(beatBufCnt>0){
    add_RR(beatBufDelay[beatBufCnt-1]-delay);
  }
  
  // Store beat
  beatBufDelay[beatBufCnt] = delay;
  // Update index
  beatBufCnt = beatBufCnt+1;
  
  
}

// Add RR interval in queue
void add_RR(int interval){ 
  int i = 0;
  for(i=RR_BUFFER_LENGTH-1;i>0;i--){
    RRBuf[i] = RRBuf[i-1];
  }
  RRBuf[0] = interval;
}

// Get following RR interval
int get_postRR(){ 
  return RRBuf[0];
}

// Get following RR interval
int get_preRR(){ 
  return RRBuf[1];
}

// Get local mean of RR interval
int get_meanRR(){ 
  int mean = 0;
  int i = 0;
  for(i = 0; i<RR_BUFFER_LENGTH; i++){
    mean = mean + RRBuf[i];
  }
  mean = mean>>RR_BUFFER_LENGTH_LOG2;
  return mean;
}

// Get local variance of RR interval
int get_varRR(int mean){ 
  int var = 0;
  int diff = 0;
  int i = 0;
  for(i = 0; i<RR_BUFFER_LENGTH; i++){
    diff = RRBuf[i]-mean;
    var = var + diff*diff;
  }
  var = var>>RR_BUFFER_LENGTH_LOG2;
  return var;
}

// Increment delays in the beat delay buffer by 1
void increment_beat_delay(){ 
  int i = 0;
  for(i=0;i<beatBufCnt;i++){
    beatBufDelay[i] = beatBufDelay[i]+1;
  }
}

// Return number of beats in beat delay buffer
int get_beat_number(){ 
  return beatBufCnt;
}

// Get oldest beat delay
uint16_t get_beat(){ 
  return beatBufDelay[0];
}

// Pop oldest beat
uint16_t pop_beat(){ 
  int i = 0;
  uint16_t delay = beatBufDelay[0];
  for(i=1; i<beatBufCnt; i++){
    beatBufDelay[i-1] = beatBufDelay[i];
  }
  beatBufCnt = beatBufCnt-1;
  return delay;
}

// Closing function
int close_buffers(){
  free(signalBuf);
  //free(signalOutBuf);
  free(beatBufDelay);
  free(RRBuf);
  return 1;
}