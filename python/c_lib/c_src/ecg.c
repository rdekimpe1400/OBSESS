/*****************************************************************************
FILE:  ecg.c
AUTHOR:	R. Dekimpe
REVISED:	06/2020

****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ecg.h"
#include "buffer.h"
#include "qrsdet.h"
#include "dwt_int.h"
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
    detection_delay = detection_delay - ((HPBUFFER_LGTH+LPBUFFER_LGTH)>>1);//add filter delay
    add_beat(detection_delay);
  }
  
  // If an old beat can be classified (post-RR interval available, post-QRS window signal available), process it
  if(get_beat_number()>1){
    if(get_beat()>SIGNAL_OUTPUT_AFTER){
      int32_t dec_values[3];
      *delay = get_beat();
      features = extract_features();
      features_select = select_features(features);
      *output=svm_predict(features_select, dec_values );
    }
  }
  
  // Increment beat delays in buffer
  increment_beat_delay();
  
  return features;
}

void ECG_init(){
  int16_t dummy;
  QRSDet( 0, 0, 1 );
  QRSFilter( 0, 1, &dummy );
  init_buffers();
  dwt_bufferinit(DWT_LENGTH, DWT_LEVEL);
  return;
}

void ECG_close(){
  close_buffers();
  return;
}
