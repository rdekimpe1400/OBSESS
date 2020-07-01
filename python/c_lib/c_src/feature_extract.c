/*****************************************************************************
FILE:  features.c
AUTHOR:	R. Dekimpe
REVISED:	06/2020

****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "buffer.h"
#include "dwt_int.h"
#include "feature_extract.h"

// Global variables
int16_t features[FEATURES_COUNT];   // Output signal buffer

const int time_idx[FEATURES_COUNT_TIME] = {-36,-33,-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,30,40,50,60,70,80};

int16_t* extract_features(void){
  int beat_delay = pop_beat();
  int16_t* signal_segment = get_segment(beat_delay);
  int dwt_outlen;
      
  int16_t* dwt = wavedec(signal_segment+(SIGNAL_OUTPUT_BEFORE-DWT_BEFORE), DWT_LENGTH, DWT_LEVEL, &dwt_outlen);
  
  features[0] = get_postRR();
  features[1] = get_preRR();
  features[2] = get_meanRR();
  features[3] = get_varRR(features[2]);
  
  for(int i=0;i<FEATURES_COUNT_TIME;i++){
    features[FEATURES_COUNT_RR+i] = signal_segment[SIGNAL_OUTPUT_BEFORE+time_idx[i]];
  }
  
  for(int i=0;i<FEATURES_COUNT_DWT;i++){
    features[FEATURES_COUNT_RR+FEATURES_COUNT_TIME+i] = dwt[FEATURES_DWT_START+i];
  }
  
  return features;
}
      