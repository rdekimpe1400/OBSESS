/*****************************************************************************
FILE:  qrsdet.h
Software Foundation; either version 2 of the License, or (at your option) any
later version.

This software is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Library General Public License for more
details.

You should have received a copy of the GNU Library General Public License along
with this library; if not, write to the Free Software Foundation, Inc., 59
Temple Place - Suite 330, Boston, MA 02111-1307, USA.

You may contact the author by e-mail (pat@eplimited.com) or postal mail
(Patrick Hamilton, E.P. Limited, 35 Medford St., Suite 204 Somerville,
MA 02143 USA).  For updates to this software, please visit our website
(http://www.eplimited.com).
  __________________________________________________________________________
  Revisions:
	4/16: Modified to allow simplified modification of digital filters in
   	qrsfilt().
*****************************************************************************/

#include <stdint.h>

//#define SKIP_INIT

#define SAMPLE_RATE	200	/* Sample rate in Hz. */
#define MS_PER_SAMPLE	5
#define MS10	2
#define MS25	5
#define MS30	6
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
#define DERIV_LENGTH	2
#define LPBUFFER_LGTH 10
#define HPBUFFER_LGTH 25

#define WINDOW_WIDTH	16
#define	FILTER_DELAY (17 + PRE_BLANK)  // filter delays plus 200 ms blanking delay
#define DER_DELAY	36 + FILTER_DELAY

#define ECG_BUFFER_LENGTH 400

#define HPBUFFER_LGTH 25

#define AMPL_NORM_SHIFT 12
#define AMPL_NORM_TARGET 5000
#define GAIN_SMOOTH_DECAY_LOG 5

int QRSDet( int datum, int fdatum, int init );
int QRSFilter(int datum,int init,int16_t* datum_filt);
int init_QRSNorm(void);
int QRSNorm(int datum);
void QRSNorm_updateGain(int amplitude);

//#define SAMPLE_RATE	200	/* Sample rate in Hz. */
//#define MS_PER_SAMPLE	( (double) 1000/ (double) SAMPLE_RATE)
//#define MS10	((int) (10/ MS_PER_SAMPLE + 0.5))
//#define MS25	((int) (25/MS_PER_SAMPLE + 0.5))
//#define MS30	((int) (30/MS_PER_SAMPLE + 0.5))
//#define MS80	((int) (80/MS_PER_SAMPLE + 0.5))
//#define MS95	((int) (95/MS_PER_SAMPLE + 0.5))
//#define MS100	((int) (100/MS_PER_SAMPLE + 0.5))
//#define MS125	((int) (125/MS_PER_SAMPLE + 0.5))
//#define MS150	((int) (150/MS_PER_SAMPLE + 0.5))
//#define MS160	((int) (160/MS_PER_SAMPLE + 0.5))
//#define MS175	((int) (175/MS_PER_SAMPLE + 0.5))
//#define MS195	((int) (195/MS_PER_SAMPLE + 0.5))
//#define MS200	((int) (200/MS_PER_SAMPLE + 0.5))
//#define MS220	((int) (220/MS_PER_SAMPLE + 0.5))
//#define MS250	((int) (250/MS_PER_SAMPLE + 0.5))
//#define MS300	((int) (300/MS_PER_SAMPLE + 0.5))
//#define MS360	((int) (360/MS_PER_SAMPLE + 0.5))
//#define MS450	((int) (450/MS_PER_SAMPLE + 0.5))
//#define MS1000	SAMPLE_RATE
//#define MS1500	((int) (1500/MS_PER_SAMPLE))
//#define DERIV_LENGTH	MS10
//#define LPBUFFER_LGTH ((int) (2*MS25))
//#define HPBUFFER_LGTH MS125
//
//#define WINDOW_WIDTH	MS80			// Moving window integration width.
//#define	FILTER_DELAY (int) (((double) DERIV_LENGTH/2) + ((double) LPBUFFER_LGTH/2 - 1) + (((double) HPBUFFER_LGTH-1)/2) + PRE_BLANK)  // filter delays plus 200 ms blanking delay