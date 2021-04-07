/*****************************************************************************
FILE:  features.c
AUTHOR:	R. Dekimpe
REVISED:	06/2020

****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "beat.h"
#include "signal_buffer.h"
#include "dwt_int.h"
#include "svm.h"
#include "feature_extract.h"

#define DEBUG_PRINT

// Global variables
int16_t features[FEATURES_COUNT];   // Output feature buffer
int16_t *features_select;   // Output selected feature buffer

int16_t **smooth_features_buffer; // Buffer 
int32_t *smooth_features_sum; // Buffer 
int smooth_idx;

const int time_idx[FEATURES_COUNT_TIME] = {-39,-36,-33,-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80};

features_t* new_features(){
  features_t* feat_struct = (features_t*) malloc(sizeof(features_t));
  
  feat_struct->prev_RR = DEFAULT_RR;
  feat_struct->next_RR = DEFAULT_RR;
  feat_struct->avrg_RR = DEFAULT_RR;
  feat_struct->time = (int16_t*) malloc(FEATURES_COUNT_TIME*sizeof(int16_t));
  
	return feat_struct;
}

int16_t* get_features(features_t* feat_struct){
  features[0] = feat_struct->prev_RR;
  features[1] = feat_struct->next_RR;
  features[2] = feat_struct->avrg_RR;
  
  memcpy(&(features[3]),feat_struct->time,FEATURES_COUNT_TIME*sizeof(int16_t));
  
#ifdef DEBUG_PRINT
  printf("Features = ");
  int i;
  for(i = 0; i<FEATURES_COUNT; i++){
    printf("%d, ",features[i]);
  }
  printf("\n");
#endif
  return features;
}

int extract_features_RR(beat_t* beat){
#ifdef DEBUG_PRINT
  printf("Extract RR\n");
#endif
  int16_t new_RR = beat->delay - beat->next_beat->delay;
  
  // New RR interval 
  beat->features->next_RR = new_RR;
  beat->next_beat->features->prev_RR = new_RR;
  
  // Past average RR interval (single pole IIR)
  beat->next_beat->features->avrg_RR = beat->features->avrg_RR - (beat->features->avrg_RR >> FEATURES_RR_SMOOTH_DECAY_LOG) + (new_RR >> FEATURES_RR_SMOOTH_DECAY_LOG);
  
  return 0;
}

int extract_features_time(beat_t* beat){
  int i;
#ifdef DEBUG_PRINT
  printf("Extract time\n");
#endif
  
  for(i=0;i<FEATURES_COUNT_TIME; i++){
    beat->features->time[i] = beat->signal[SIGNAL_SEGMENT_BEFORE+time_idx[i]];
  }
  //memcpy(beat->features->time,&(beat->signal[SIGNAL_SEGMENT_BEFORE-10]),FEATURES_COUNT_TIME*sizeof(int16_t));
  
#ifdef DEBUG_PRINT
  printf("    = ");
  for(i = 0; i<FEATURES_COUNT_TIME; i++){
    printf("%d, ",beat->features->time[i]);
  }
  printf("\n");
#endif
  return 0;
}

int16_t* select_features(int16_t* features_all){
  int i = 0;
  for(i=0; i<n_feat; i++){
    features_select[i] = features_all[feature_select_idx[i]];
  }
  
  return features_select;
}

int init_features_buffer(){   
  int i = 0;
  for(i=0; i<FEATURES_COUNT; i++){
    features[i]=0;
  }
  return 0;
}

int16_t* get_features_buffer(){
  return features;
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

int delete_features(features_t* feat_struct){
  free(feat_struct->time);
  free(feat_struct);
  return 0;
}