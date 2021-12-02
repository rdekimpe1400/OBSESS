
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "qrsdet.h"

// Local Prototypes.
//int gain( int datum ,int gain) ;

int g = 4096;


int init_QRSNorm()
	{
	g = 4096;
  return 0;
	}
  
int QRSNorm(int datum)
	{
	int ndatum ;
	ndatum = (datum*g)>>AMPL_NORM_SHIFT;
	return(ndatum) ;
	}

void QRSNorm_updateGain(int amplitude)
{
	int new_g;
  new_g = (AMPL_NORM_TARGET*g)/(amplitude+1); // Compute ideal gain based on latest amplitude and current gain
  
  g = g - (g >> GAIN_SMOOTH_DECAY_LOG) + (new_g >> GAIN_SMOOTH_DECAY_LOG); // single pole IIR
}