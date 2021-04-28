/*****************************************************************************
FILE:  features.c
AUTHOR:	R. Dekimpe
REVISED:	06/2020

****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "dwt.h"
#include "beat.h"
#include "signal_buffer.h"
#include "svm.h"
#include "feature_extract.h"

//#define DEBUG_PRINT

// Global variables
int16_t *features_buffer;   // Output feature buffer
int16_t *features_select;   // Output selected feature buffer

int16_t **smooth_features_buffer; // Buffer 
int32_t *smooth_features_sum; // Buffer 
int smooth_idx;

const int time_idx[FEATURES_COUNT_TIME] = {-39,-36,-33,-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80};

int16_t* dwt_template_N;
int template_init;

features_t* new_features(){
  features_t* features = (features_t*) malloc(sizeof(features_t));
  
  features->prev2_RR = DEFAULT_RR;
  features->prev_RR = DEFAULT_RR;
  features->next_RR = DEFAULT_RR;
  features->avrg_RR = DEFAULT_RR;
  features->prev_diff_RR = 0;
  features->next_diff_RR = 0;
  features->ratio1_RR = 1<<FEATURES_RR_RATIO_SHIFT;
  features->ratio2_RR = 1<<FEATURES_RR_RATIO_SHIFT;
  
  features->time_len = SIGNAL_SEGMENT_LENGTH;
  features->time = (int16_t*) malloc(features->time_len*sizeof(int16_t));
  features->dwt_len = dwt_bufferlen(DWT_LENGTH, DWT_LEVEL);
  features->dwt = (int16_t*) malloc(features->dwt_len*sizeof(int16_t));
  features->dwt_diff = (int16_t*) malloc(features->dwt_len*sizeof(int16_t));
  
	return features;
}

int init_features_buffer(){   
  int i = 0;
  features_buffer = (int16_t*) malloc(FEATURES_COUNT*sizeof(int16_t));
  for(i=0; i<FEATURES_COUNT; i++){
    features_buffer[i]=0;
  }
  
  int dwt_len = dwt_bufferlen(DWT_LENGTH, DWT_LEVEL);
  dwt_template_N = (int16_t*) malloc(dwt_len*sizeof(int16_t));
  for(i=0; i<dwt_len; i++){
    dwt_template_N[i]=0;
  }
  template_init = 0;
  
  return 0;
}

int16_t* get_features(beat_t* beat){
  
  gather_features(beat, features_buffer);
  
#ifdef DEBUG_PRINT
  printf("Features = ");
  int i;
  for(i = 0; i<FEATURES_COUNT; i++){
    printf("%d, ",features_buffer[i]);
  }
  printf("\n");
#endif
  return features_buffer;
}

int extract_features_RR(beat_t* beat){
#ifdef DEBUG_PRINT
  printf("Extract RR (%d) \n", FEATURES_COUNT_RR);
#endif
  int16_t new_RR = beat->delay - beat->next_beat->delay;
  
  // New RR interval 
  beat->features->next_RR = new_RR;
  beat->next_beat->features->prev_RR = new_RR;
  beat->next_beat->features->prev2_RR = beat->features->prev_RR;
  
  // Past average RR interval (single pole IIR)
  beat->next_beat->features->avrg_RR = beat->features->avrg_RR - (beat->features->avrg_RR >> FEATURES_SMOOTH_DECAY_LOG) + (new_RR >> FEATURES_SMOOTH_DECAY_LOG);
  
  // Compute diff and ratio
  beat->features->prev_diff_RR = beat->features->prev_RR - beat->features->avrg_RR;
  beat->features->next_diff_RR = beat->features->next_RR - beat->features->avrg_RR;
  beat->features->ratio1_RR = (beat->features->prev_RR<<FEATURES_RR_RATIO_SHIFT)/beat->features->prev2_RR;
  beat->features->ratio2_RR = (beat->features->next_RR<<FEATURES_RR_RATIO_SHIFT)/beat->features->prev_RR;
  
  return 0;
}

int extract_features_time(beat_t* beat){
#ifdef DEBUG_PRINT
  printf("Extract time (%d) \n", SIGNAL_SEGMENT_LENGTH);
#endif

  memcpy(beat->features->time, beat->signal, SIGNAL_SEGMENT_LENGTH*sizeof(int16_t));
  
#ifdef DEBUG_PRINT
  int j;
  printf("    = ");
  for(j = 0; j<SIGNAL_SEGMENT_LENGTH; j++){
    printf("%d, ",beat->features->time[j]);
  }
  printf("\n");
#endif
  return 0;
}

int extract_features_DWT(beat_t* beat){
  int i;
#ifdef DEBUG_PRINT
  printf("Extract DWT (%d) \n", beat->features->dwt_len);
#endif
  
  wavedec(&(beat->signal[SIGNAL_SEGMENT_BEFORE-DWT_BEFORE]), DWT_LENGTH, DWT_LEVEL, beat->features->dwt);
  
  for(i=0; i<beat->features->dwt_len ; i++){
    beat->features->dwt_diff[i] = beat->features->dwt[i] - dwt_template_N[i];
  }
  
#ifdef DEBUG_PRINT
  int j;
  printf("    = ");
  for(j = 0; j<FEATURES_COUNT_TIME; j++){
    printf("%d, ",beat->features->dwt[j]);
  }
  printf("\n");
#endif
  return 0;
}

int gather_features(beat_t* beat, int16_t* features_out){
  int i;
  
  features_out[0] = beat->features->prev_RR;
  features_out[1] = beat->features->next_RR;
  features_out[2] = beat->features->prev2_RR;
  features_out[3] = beat->features->avrg_RR;
  features_out[4] = beat->features->prev_diff_RR;
  features_out[5] = beat->features->next_diff_RR;
  features_out[6] = beat->features->ratio1_RR;
  features_out[7] = beat->features->ratio2_RR;
  
  for(i=0; i<FEATURES_COUNT_TIME; i++){
    features_out[FEATURES_COUNT_RR+i] = beat->features->time[SIGNAL_SEGMENT_BEFORE+time_idx[i]];
  }
  
  for(i=0; i<FEATURES_COUNT_DWT; i++){
    features_out[FEATURES_COUNT_RR+FEATURES_COUNT_TIME+i] = beat->features->dwt_diff[FEATURES_DWT_START+i];
  }
  
  return FEATURES_COUNT;
}

int update_feature_template(beat_t* beat){ 
  int i;
  if((beat->gold_label==1)||(beat->gold_label==2)){
    if(template_init){
      for(i=0; i<beat->features->dwt_len; i++){
        dwt_template_N[i] = dwt_template_N[i] - (dwt_template_N[i] >> FEATURES_SMOOTH_DECAY_LOG) + (beat->features->dwt[i] >> FEATURES_SMOOTH_DECAY_LOG);
      }
    }else{
      for(i=0; i<beat->features->dwt_len; i++){
        dwt_template_N[i] = beat->features->dwt[i];
      }
      template_init = 1;
    }
  }
  return 0;
}

int16_t* get_features_buffer(){
  return features_buffer;
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

int delete_features(features_t* features){
  free(features->time);
  free(features->dwt);
  free(features);
  return 0;
}