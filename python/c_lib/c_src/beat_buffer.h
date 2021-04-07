//**********************************************
//
//  ECG buffer management
//
//**********************************************

#ifndef BEATBUF_H_
#define BEATBUF_H_

#include <stdint.h>

// Parameter definition
#define BEAT_BUFFER_LENGTH      10

// Function declaration
int init_beat_buffer(void);
int close_beat_buffer(void);
void add_beat(int16_t delay);
void increment_beat_delay(void);
uint16_t pop_beat();
int is_beat_ready(void);
int16_t* buffer_get_features(void);

#endif //BEATBUF_H_
