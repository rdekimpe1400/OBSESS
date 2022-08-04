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
#include "upsample.h" 

int16_t* ECG_wrapper( int sample, int label_gold, int* delay, int* output, int* gold_label){
  int ecg_transform = 0;
  int ecg = sample;
  int16_t ecg_filt = 0;
  int detection_delay = 0;
  int16_t* features = 0;
  int amplitude = 0;
  int16_t* upsample_buffer;
  int upsample_buffer_length;
  int i = 0;
  
  *output = 0;
  *delay = 0;
  
  ecg=QRSNorm(ecg);
  
  // QRS beat enhancement for detection
  ecg_transform=QRSFilter(ecg, 0,&ecg_filt );
  
  // Upsample
  upsample_buffer = upsample(ecg_filt, &upsample_buffer_length);
  
  // Put new sample in signal buffer
  for(i = 0; i<upsample_buffer_length; i++){
    push_sample(upsample_buffer[i], label_gold);
  }
  
  // QRS detection
  detection_delay=QRSDet(ecg, ecg_transform,0 ) ; 
  
  // If beat is detected, add it in beat buffer
  if(detection_delay){
    //detection_delay = detection_delay;// - ((HPBUFFER_LGTH+LPBUFFER_LGTH)>>1);//add filter delay
    detection_delay = (detection_delay<<LOG2_N)/M;
    add_beat(detection_delay);
  }
  
  // If an old beat can be classified (post-RR interval available, post-QRS window signal available), process it
  if(is_beat_ready()){
    features = buffer_get_features(&amplitude);
    //*delay = pop_beat(gold_label);
    *delay = (pop_beat(gold_label)*M)>>LOG2_N;
    //features = extract_features();
    //features_select = select_features(features);  
    *output=svm_predict(features );
    
    QRSNorm_updateGain(amplitude);
  }
  
  // Increment beat delays in buffer
  increment_beat_delay(upsample_buffer_length);
  
  return features;
}

void ECG_init(){
  int16_t dummy;
  QRSDet( 0, 0, 1 );
  QRSFilter( 0, 1, &dummy );
  init_QRSNorm();
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
  close_features_buffer();
  return;
}

