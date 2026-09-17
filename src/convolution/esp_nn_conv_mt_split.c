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

typedef void (*esp_nn_conv_s8_target_fn_t)(const data_dims_t *,
                                           const int8_t *,
                                           const data_dims_t *,
                                           const int8_t *,
                                           const int32_t *,
                                           const data_dims_t *,
                                           int8_t *,
                                           const conv_params_t *,
                                           const quant_data_t *);

/* Grouped convolution as G standard convolutions through the target's
 * optimized path. Each group g reads input channels [g*filter_ch, ...) and
 * writes output channels [g*fpg, ...): the group's input slice is repacked
 * contiguous into scratch, the group's filters are already contiguous, the
 * group's output is staged contiguous and scattered back. Arithmetic per
 * output element is identical to the reference grouped loop (bit-exact,
 * given the target conv is reference-exact). Returns false (nothing
 * written) when shapes or scratch don't fit. */
bool esp_nn_conv_s8_grouped_repack(esp_nn_conv_s8_target_fn_t conv_fn,
                                   const data_dims_t *input_dims,
                                   const int8_t *input_data,
                                   const data_dims_t *filter_dims,
                                   const int8_t *filter_data,
                                   const int32_t *bias,
                                   const data_dims_t *output_dims,
                                   int8_t *out_data,
                                   const conv_params_t *conv_params,
                                   const quant_data_t *quant_data,
                                   void *scratch, int scratch_size,
                                   int inner_scratch_size)
{
    const int32_t in_ch = input_dims->channels;
    const int32_t filter_ch = filter_dims->channels;
    const int32_t out_ch = output_dims->channels;
    if (filter_ch <= 0 || in_ch % filter_ch != 0) {
        return false;
    }
    const int32_t groups = in_ch / filter_ch;
    if (groups <= 1 || out_ch % groups != 0) {
        return false;
    }
    /* Per-group filter rows narrower than one 16-byte SIMD load leave most
     * lanes idle and the repack copies dominate (e.g. depthwise-as-grouped,
     * filter_ch == 1): the reference grouped loop is faster there. */
    if ((int32_t)filter_dims->width * filter_ch < 16) {
        return false;
    }
    const int32_t fpg = out_ch / groups;
    const int32_t pixels_in = (int32_t)input_dims->width * input_dims->height;
    const int32_t pixels_out = (int32_t)output_dims->width * output_dims->height;
    const int32_t in_slice_bytes = pixels_in * filter_ch;
    const int32_t out_slice_bytes = pixels_out * fpg;
    const int32_t need = inner_scratch_size + in_slice_bytes + out_slice_bytes + 32;
    if (scratch == NULL || scratch_size < need) {
        return false;
    }
    int8_t *in_slice = (int8_t *)(((uintptr_t)((int8_t *)scratch + scratch_size
                                   - in_slice_bytes - out_slice_bytes - 16)) & ~(uintptr_t)15);
    int8_t *out_slice = in_slice + in_slice_bytes;

    data_dims_t in_g = *input_dims;
    data_dims_t filt_g = *filter_dims;
    data_dims_t out_g = *output_dims;
    in_g.channels = filter_ch;
    out_g.channels = fpg;
    const int32_t filter_block = fpg * filter_ch *
            (int32_t)filter_dims->width * filter_dims->height;

    for (int32_t g = 0; g < groups; g++) {
        /* Gather this group's input channels contiguously. */
        const int8_t *src = input_data + g * filter_ch;
        int8_t *dst = in_slice;
        for (int32_t p = 0; p < pixels_in; p++) {
            memcpy(dst, src, filter_ch);
            dst += filter_ch;
            src += in_ch;
        }
        quant_data_t q_g = *quant_data;
        q_g.shift = quant_data->shift + g * fpg;
        q_g.mult = quant_data->mult + g * fpg;
        conv_fn(&in_g, in_slice, &filt_g, filter_data + g * filter_block,
                bias ? bias + g * fpg : NULL, &out_g, out_slice,
                conv_params, &q_g);
        /* Scatter the staged output into the group's channel columns. */
        const int8_t *osrc = out_slice;
        int8_t *odst = out_data + g * fpg;
        for (int32_t p = 0; p < pixels_out; p++) {
            memcpy(odst, osrc, fpg);
            osrc += fpg;
            odst += out_ch;
        }
    }
    return true;
}
