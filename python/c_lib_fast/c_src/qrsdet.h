
#include <stdint.h>

//#define SKIP_INIT

#define SAMPLE_RATE	200	/* Sample rate in Hz. */
#define MS10	2
#define MS25	5
#define MS30	6
#define MS50	10
#define MS80	16
#define MS95	19
#define MS100	20
#define MS125	25
#define MS150	30
#define MS160	32
#define MS175	35
#define MS195	39
#define MS200	40
#define MS220	44
#define MS250	50
#define MS300	60
#define MS360	72
#define MS450	90
#define MS1000	200
#define MS1500	300
#define DERIV_LENGTH	MS10
#define LPBUFFER_LGTH 10
#define HPBUFFER_LGTH MS125

#define WINDOW_WIDTH	16
#define	FILTER_DELAY (18+PRE_BLANK)
#define DER_DELAY	WINDOW_WIDTH + FILTER_DELAY + MS100

#define ECG_BUFFER_LENGTH 400

#define AMPL_NORM_SHIFT 12
#define AMPL_NORM_TARGET 5000
#define GAIN_SMOOTH_DECAY_LOG 5

int QRSDet( int datum, int fdatum, int init );
int QRSFilter(int datum,int init,int16_t* datum_filt);
int init_QRSNorm(void);
int QRSNorm(int datum);
void QRSNorm_updateGain(int amplitude);
