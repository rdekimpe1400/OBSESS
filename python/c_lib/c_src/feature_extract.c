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
#include "svm.h"
#include "feature_extract.h"

// Global variables
int16_t features[FEATURES_COUNT];   // Output feature buffer
int16_t *features_select;   // Output selected feature buffer

int16_t **smooth_features_buffer; // Buffer 
int32_t *smooth_features_sum; // Buffer 
int smooth_idx;

const int time_idx[FEATURES_COUNT_TIME] = {-36,-33,-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,30,40,50,60,70,80};

int16_t* extract_features(void){
  int beat_delay = pop_beat();
  int16_t* signal_segment = get_segment(beat_delay);
  int dwt_outlen;
  int i = 0;
      
  int16_t* dwt = wavedec(signal_segment+(SIGNAL_OUTPUT_BEFORE-DWT_BEFORE), DWT_LENGTH, DWT_LEVEL, &dwt_outlen);  
  
  features[0] = get_postRR();
  features[1] = get_preRR();
  features[2] = get_meanRR();
  features[3] = get_varRR(features[2]);
  
  features[0] = (1000*features[0])/features[2];
  features[1] = (1000*features[1])/features[2];
  
  for(i=0;i<FEATURES_COUNT_TIME;i++){
    features[FEATURES_COUNT_RR+i] = signal_segment[SIGNAL_OUTPUT_BEFORE+time_idx[i]];
  }
  
  for(i=0;i<FEATURES_COUNT_DWT;i++){
    features[FEATURES_COUNT_RR+FEATURES_COUNT_TIME+i] = dwt[FEATURES_DWT_START+i];
  }
  
  //deviation_features((features+FEATURES_COUNT_RR), FEATURES_COUNT_TIME+FEATURES_COUNT_DWT);
  
  return features;
}

int16_t* select_features(int16_t* features_all){
  int i = 0;
  for(i=0; i<n_feat; i++){
    features_select[i] = features_all[feature_select_idx[i]];
  }
  
  return features_select;
}

// Compute local mean of features (using rolling window average and data buffer)
void smooth_features_init(int length){
  int i = 0;
  int j = 0;
  // Init data buffer
  smooth_features_buffer = (int16_t **)malloc(length * sizeof(int16_t *)); 
  for (i=0; i<length; i++) {
    smooth_features_buffer[i] = (int16_t *)malloc(FEATURES_SMOOTH_COUNT * sizeof(int16_t)); 
    for(j=0; j<FEATURES_SMOOTH_COUNT; j++){
      smooth_features_buffer[i][j] = 0;
    }
  }
  // Init sum buffer
  smooth_features_sum = (int32_t *)malloc(length * sizeof(int32_t)); 
  for(i=0; i<length; i++){
    smooth_features_sum[i] = 0;
  }
  // Init index
  smooth_idx = 0;
  // Init feature select buffer
  features_select = (int16_t *)malloc(n_feat * sizeof(int16_t)); 
  for(i=0; i<n_feat; i++){
    features_select[i] = 0;
  }
}

// Compute deviation from local average
int16_t* deviation_features(int16_t* input_features, int length){
  int16_t out_feature;
  int i = 0;
  for (i=0; i<length; i++) {
    // Compute output
    out_feature = smooth_features_sum[i]>>FEATURES_SMOOTH_COUNT_LOG;
    // Update sum and data buffer
    smooth_features_sum[i] = smooth_features_sum[i] + input_features[i] - smooth_features_buffer[i][smooth_idx];
    smooth_features_buffer[i][smooth_idx] = input_features[i];
    // Return difference between input and smooth average
    input_features[i] = input_features[i]-out_feature;
  }
  
  // Increment index
  smooth_idx++;
  if(smooth_idx == FEATURES_SMOOTH_COUNT){
    smooth_idx = 0;
  }
  
  return input_features;
}
