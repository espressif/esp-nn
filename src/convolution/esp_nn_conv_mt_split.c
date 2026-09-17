/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Dual-core output-row split around the reference conv implementation.
 *
 * Used by the target dispatchers for shapes that fall back to
 * esp_nn_conv_s8_ansi (e.g. grouped convolutions). The top slice keeps the
 * top padding; the bottom slice enters the input at an interior row
 * (h0 * stride_ht - pad_ht >= 0) with pad_ht = 0, so both slices evaluate
 * exactly the taps the single call would and write disjoint output rows:
 * results are bit-identical. Falls through to the plain reference call
 * when the worker is not running.
 */

#include <string.h>

#include <esp_nn_defs.h>
#include <esp_nn_ansi_headers.h>
#include <esp_nn_multicore.h>

typedef struct {
    data_dims_t input_dims;
    const int8_t *input_data;
    data_dims_t filter_dims;
    const int8_t *filter_data;
    const int32_t *bias;
    data_dims_t output_dims;
    int8_t *out_data;
    conv_params_t conv_params;
    quant_data_t quant_data;
} conv_ansi_mt_job_t;

static void conv_ansi_mt_worker(void *p)
{
    const conv_ansi_mt_job_t *j = (const conv_ansi_mt_job_t *)p;
    esp_nn_conv_s8_ansi(&j->input_dims, j->input_data, &j->filter_dims,
                        j->filter_data, j->bias, &j->output_dims, j->out_data,
                        &j->conv_params, &j->quant_data);
}

void esp_nn_conv_s8_ansi_mt_split(const data_dims_t *input_dims,
                                  const int8_t *input_data,
                                  const data_dims_t *filter_dims,
                                  const int8_t *filter_data,
                                  const int32_t *bias,
                                  const data_dims_t *output_dims,
                                  int8_t *out_data,
                                  const conv_params_t *conv_params,
                                  const quant_data_t *quant_data)
{
    const uint16_t out_ht = output_dims->height;
    const uint16_t h0 = out_ht / 2;
    const int32_t in_row_off =
            (int32_t)h0 * conv_params->stride.height - conv_params->padding.height;

    if (esp_nn_dual_core_active() && out_ht >= 4 && in_row_off >= 0) {
        conv_ansi_mt_job_t job = {
            .input_dims = *input_dims, .input_data = input_data,
            .filter_dims = *filter_dims, .filter_data = filter_data,
            .bias = bias, .output_dims = *output_dims, .out_data = out_data,
            .conv_params = *conv_params, .quant_data = *quant_data,
        };
        job.output_dims.height = h0;
        if (esp_nn_dual_core_run(conv_ansi_mt_worker, &job)) {
            data_dims_t in1 = *input_dims;
            data_dims_t outd1 = *output_dims;
            conv_params_t params1 = *conv_params;
            in1.height = input_dims->height - in_row_off;
            outd1.height = out_ht - h0;
            params1.padding.height = 0;
            esp_nn_conv_s8_ansi(
                    &in1,
                    input_data + in_row_off * input_dims->width * input_dims->channels,
                    filter_dims, filter_data, bias, &outd1,
                    out_data + (int32_t)h0 * output_dims->width * output_dims->channels,
                    &params1, quant_data);
            esp_nn_dual_core_wait();
            return;
        }
    }
    esp_nn_conv_s8_ansi(input_dims, input_data, filter_dims, filter_data,
                        bias, output_dims, out_data, conv_params, quant_data);
}
