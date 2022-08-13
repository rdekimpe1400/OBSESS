
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "upsample.h"

int16_t upSampleBuf[40];

int16_t* upsample(int16_t sample, int* length)
{
  static int idx = 0;
  static int16_t prev_sample = 0;
  
  if(N==1){
    *length = 1;
    upSampleBuf[0] = sample;
  }else{
    idx = idx + N;
    *length = 0;
    
    while(idx>=M){
      idx = idx-M;
      upSampleBuf[*length] = sample - ((((int32_t)(sample-prev_sample))*idx)>>LOG2_N);
      *length = *length+1;
    }
    prev_sample = sample;
  }
  return upSampleBuf;
}
  