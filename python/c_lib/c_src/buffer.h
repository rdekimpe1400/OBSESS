//**********************************************
//
//  ECG buffer management
//
//**********************************************

#ifndef BUF_H_
#define BUF_H_

#include <stdint.h>

// Parameter definition
#define SAMPLING_FREQUENCY      200
#define SIGNAL_BUFFER_DURATION  5 
#define SIGNAL_BUFFER_LENGTH    SAMPLING_FREQUENCY*SIGNAL_BUFFER_DURATION
#define SIGNAL_OUTPUT_BEFORE    100
#define SIGNAL_OUTPUT_AFTER     100
#define SIGNAL_OUTPUT_LENGTH    SIGNAL_OUTPUT_BEFORE+SIGNAL_OUTPUT_AFTER
#define BEAT_BUFFER_LENGTH      3
#define RR_BUFFER_LENGTH        16
#define RR_BUFFER_LENGTH_LOG2   4
#define DEFAULT_RR              1*SAMPLING_FREQUENCY

// Function declaration
int init_buffers();
int close_buffers();
void push_sample(int16_t sample);
int16_t* get_segment(int delay);
void add_beat(int16_t delay);
void increment_beat_delay();
int get_beat_number();
uint16_t get_beat();
uint16_t pop_beat();
void add_RR(int interval);
int get_postRR(void);
int get_preRR(void);
int get_meanRR(void);
int get_varRR(int mean);

#endif //BUF_H_
