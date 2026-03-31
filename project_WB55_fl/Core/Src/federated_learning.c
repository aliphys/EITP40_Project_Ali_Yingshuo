/* federated_learning.c */
#include "federated_learning.h"
#include "config.h"
#include "nn.h"
#include "nn_ff.h"
#include <string.h>

#define FL_STATE_BYTES_MAX 32768
#define FL_STATE_WORDS_MAX (FL_STATE_BYTES_MAX / sizeof(float))

static fl_role_t fl_role = FL_ROLE_UNKNOWN;
static uint32_t fl_expected_clients = 0;
static uint32_t fl_update_count = 0;
static uint8_t fl_has_aggregated = 0;

static float fl_accumulator[FL_STATE_WORDS_MAX];
static float fl_aggregated[FL_STATE_WORDS_MAX];

size_t fl_state_size(void)
{
#if USE_BACKPROP
    return nn_state_size();
#elif USE_FF
    return nn_ff_state_size();
#else
    return 0;
#endif
}

static size_t fl_state_words(void)
{
    return fl_state_size() / sizeof(float);
}

void fl_init(fl_role_t role, uint32_t expected_clients)
{
    fl_role = role;
    fl_expected_clients = expected_clients;
    fl_update_count = 0;
    fl_has_aggregated = 0;
    memset(fl_accumulator, 0, sizeof(fl_accumulator));
    memset(fl_aggregated, 0, sizeof(fl_aggregated));
}

void fl_set_role(fl_role_t role)
{
    fl_role = role;
}

fl_role_t fl_get_role(void)
{
    return fl_role;
}

uint8_t fl_is_server(void)
{
    return (fl_role == FL_ROLE_SERVER) ? 1 : 0;
}

uint8_t fl_is_client(void)
{
    return (fl_role == FL_ROLE_CLIENT) ? 1 : 0;
}

void fl_reset_accumulator(void)
{
    fl_update_count = 0;
    fl_has_aggregated = 0;
    memset(fl_accumulator, 0, sizeof(fl_accumulator));
}

int fl_accumulate_weights(const void *state, size_t len)
{
    if (!fl_is_server() || state == NULL)
        return -1;

    size_t expected = fl_state_size();
    if (len != expected)
        return -2;

    size_t words = fl_state_words();
    const float *values = (const float *)state;

    if (fl_update_count == 0)
    {
        memset(fl_accumulator, 0, words * sizeof(float));
    }

    for (size_t i = 0; i < words; i++)
    {
        fl_accumulator[i] += values[i];
    }

    fl_update_count++;
    return 0;
}

int fl_commit_aggregation(void)
{
    if (!fl_is_server())
        return -1;

    if (fl_update_count == 0)
        return -2;

    size_t words = fl_state_words();
    float scale = 1.0f / (float)fl_update_count;

    for (size_t i = 0; i < words; i++)
    {
        fl_aggregated[i] = fl_accumulator[i] * scale;
        // Keep accumulator for diagnostics
    }

#if USE_BACKPROP
    if (!nn_state_import((const void *)fl_aggregated, words * sizeof(float)))
    {
        return -3;
    }
#elif USE_FF
    if (!nn_ff_state_import((const void *)fl_aggregated, words * sizeof(float)))
    {
        return -3;
    }
#endif

    fl_has_aggregated = 1;
    return 0;
}

int fl_get_aggregated_state(void *dst, size_t len)
{
    if (dst == NULL || !fl_has_aggregated)
        return -1;

    size_t expected = fl_state_size();
    if (len != expected)
        return -2;

    memcpy(dst, (const void *)fl_aggregated, len);
    return 0;
}

uint32_t fl_client_update_count(void)
{
    return fl_update_count;
}
