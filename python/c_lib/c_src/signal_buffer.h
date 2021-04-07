//**********************************************
//
//  ECG buffer management
//
//**********************************************

#ifndef SIGBUF_H_
#define SIGBUF_H_

#include <stdint.h>

// Parameter definition
#define SAMPLING_FREQUENCY      200
#define SIGNAL_BUFFER_DURATION  5 
#define SIGNAL_BUFFER_LENGTH    (SAMPLING_FREQUENCY*SIGNAL_BUFFER_DURATION)
#define SIGNAL_SEGMENT_BEFORE    100
#define SIGNAL_SEGMENT_AFTER     100
#define SIGNAL_SEGMENT_LENGTH    (SIGNAL_SEGMENT_BEFORE+SIGNAL_SEGMENT_AFTER)

// Function declaration
int init_signal_buffer(void);
int close_signal_buffer(void);
void push_sample(int16_t sample);
int16_t* get_segment(int delay);

#endif //SIGBUF_H_
