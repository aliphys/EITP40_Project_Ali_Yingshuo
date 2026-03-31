/* federated_learning.h */
#pragma once

#include <stdint.h>
#include <stddef.h>

typedef enum {
    FL_ROLE_UNKNOWN = 0,
    FL_ROLE_SERVER = 1,
    FL_ROLE_CLIENT = 2
} fl_role_t;

void fl_init(fl_role_t role, uint32_t expected_clients);
void fl_set_role(fl_role_t role);
fl_role_t fl_get_role(void);
uint8_t fl_is_server(void);
uint8_t fl_is_client(void);

void fl_reset_accumulator(void);
int fl_accumulate_weights(const void *state, size_t len);
int fl_commit_aggregation(void);
int fl_get_aggregated_state(void *dst, size_t len);
size_t fl_state_size(void);
uint32_t fl_client_update_count(void);
