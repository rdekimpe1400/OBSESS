/*****************************************************************************
FILE:  ecg.c
AUTHOR:	R. Dekimpe
REVISED:	06/2020

****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ecg.h"
#include "signal_buffer.h"
#include "beat_buffer.h"
#include "qrsdet.h"
#include "feature_extract.h" 
#include "svm.h"

int16_t* ECG_wrapper( int sample, int* delay, int* output){
  int ecg_transform = 0;
  int ecg = sample;
  int16_t ecg_filt = 0;
  int detection_delay = 0;
  int16_t* features = 0;
  int16_t* features_select = 0; 
  
  *output = 0;
  *delay = 0;
  
  // QRS beat enhancement for detection
  ecg_transform=QRSFilter(ecg, 0,&ecg_filt );
  
  // Put new sample in signal buffer
  push_sample(ecg_filt);
  
  // QRS detection
  detection_delay=QRSDet(ecg, ecg_transform,0 ) ; 
  
  // If beat is detected, add it in beat buffer
  if(detection_delay){
    detection_delay = detection_delay;// - ((HPBUFFER_LGTH+LPBUFFER_LGTH)>>1);//add filter delay
    add_beat(detection_delay);
  }
  
  // If an old beat can be classified (post-RR interval available, post-QRS window signal available), process it
  if(is_beat_ready()){
    int32_t dec_values[3];
    features = buffer_get_features();
    *delay = pop_beat();
    //features = extract_features();
    //features_select = select_features(features);  
    //*output=svm_predict(features_select, dec_values );
    *output = 1; 
  }
  
  
  // Increment beat delays in buffer
  increment_beat_delay();
  
  return features;
}

void ECG_init(){
  int16_t dummy;
  QRSDet( 0, 0, 1 );
  QRSFilter( 0, 1, &dummy );
  init_signal_buffer();
  init_beat_buffer();
  init_features_buffer();
  //int dwt_len = dwt_bufferinit(DWT_LENGTH, DWT_LEVEL);
  //smooth_features_init(FEATURES_COUNT_TIME+FEATURES_COUNT_DWT); 
  return;
}

void ECG_close(){
  close_beat_buffer();
  close_signal_buffer();
  return;
}

