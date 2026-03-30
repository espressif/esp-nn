/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Optimizations strategies used:
 * Below optimizations are capable of any size of input/filter:
 *
 * 1. For filter wdxht = 1x1 (Refer esp_nn_conv_s8_mult8_1x1_esp32s3 function)
 *      - For this specific version, the strategy we employ:
 *          > This particular filter has only the channel
 *              dimension and we have `out_ch` number of such filters.
 *          > We take 8 input lines at a time and transpose those.
 *          > Keep loading and multiplying filter values one by one,
 *              to produce 8 outputs in parallel
 *
 * 2. General version: (Refer esp_nn_conv_s8_filter_aligned_input_padded_esp32s3)
 *      - For all other cases:
 *          > Consider `filter_wd * in_ch` as a single row. These many values can
 *              be continuosly loaded from inputs as well.
 *          > multiply accumulate into a single filter output.
 *          > To speed things up further, we pre-calculate
 *              (filter * in_offset + bias term) earlier and add it at the end of filter
 *
 *      About ((filter * in_offset + bias term)) accumulate term:
 *          > The conv operation before requantization is as follows:
 *              for i in filter_size:
 *                  conv_out += (input + input_offset) * filter;
 *               conv_out += bias
 *
 *          > where input_offset is constant term hence, we can see that
 *              this term can be precalculated as:
 *                  for i in filter_size:
 *                      acc_term += input_offset * filter[i];
 *                  acc_term += bias
 *              OR
 *                   for i in filter_size:
 *                      acc_term += filter[i]; // accumulate filter values
 *                  acc_term = acc_term * input_offset + bias
 *
 *
 * In both the above versions we align the filter if needed, pad the input with
 *       -input_offset if needed and extend the channels to make those multiple
 *       of 8/16 as per function needs
 *
 * 3. Im2col version: (for small in_ch where filter_wd * in_ch < 16)
 *      - Inspired by ESP32-P4 im2col approach.
 *      - Instead of padding channels (wastes 81% of SIMD lanes for in_ch=3),
 *        flatten the entire filter window into one contiguous vector:
 *          window_len = filter_wd * filter_ht * in_ch (e.g., 3*3*3 = 27)
 *      - For each output pixel: copy the input window into a scratch buffer,
 *        then use ACCX dot product on the full window. No wasted MACs.
 */

#include <stdio.h>
#include <string.h>
#include <esp_nn_defs.h>

#include <common_functions.h>

static int16_t *scratch_buffer = NULL;

extern void esp_nn_conv_s8_mult8_1x1_esp32s3(
                const int8_t *input_data,
                const uint16_t input_wd,
                const uint16_t input_ht,
                const uint16_t in_channels,
                const int32_t input_offset,
                const int8_t *filter_aligned,
                const int32_t *bias,
                int8_t *out_data,
                const uint16_t out_wd,
                const uint16_t out_ht,
                const uint16_t out_channels,
                const int32_t out_offset,
                const int32_t *out_shift,
                const int32_t *out_mult,
                const int32_t activation_min,
                const int32_t activation_max,
                void *buffer /* scratch buffer */);

extern void esp_nn_conv_s8_filter_aligned_input_padded_esp32s3(
                const int8_t *input_data,
                const uint16_t input_wd,
                const uint16_t input_ht,
                const uint16_t in_channels,
                const int32_t input_offset,
                const uint16_t stride_wd,
                const uint16_t stride_ht,
                const int8_t *filter_data,
                const uint16_t filter_wd,
                const uint16_t filter_ht,
                const int32_t *bias,
                int8_t *out_data,
                const uint16_t out_wd,
                const uint16_t out_ht,
                const uint16_t out_channels,
                const int32_t out_offset,
                const int32_t *out_shift,
                const int32_t *out_mult,
                const int32_t activation_min,
                const int32_t activation_max,
                void *scratch_buffer);

/* Use shared dot product from common — see esp_nn_dot_s8_esp32s3.S */

/**
 * Im2col convolution for small in_ch (filter_wd * in_ch < 16).
 *
 * Instead of padding channels to 16 (wasting 81% MACs for in_ch=3),
 * flatten the entire filter window into one contiguous vector:
 *   window_len = filter_wd * filter_ht * in_ch (e.g., 3*3*3 = 27)
 *
 * For each output pixel: copy the input window into a contiguous scratch
 * buffer, then use ACCX dot product. No wasted MACs.
 *
 * Scratch layout: [filter_sum[out_ch] | im2col_buf[window_len_aligned]]
 */
__attribute__ ((noinline))
static void esp_nn_conv_s8_im2col_s3(
        const data_dims_t *input_dims,
        const int8_t *input_data,
        const data_dims_t *filter_dims,
        const int8_t *filter_data,
        const int32_t *bias,
        const data_dims_t *output_dims,
        int8_t *out_data,
        const conv_params_t *conv_params,
        const quant_data_t *quant_data,
        void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    const int32_t window_len = filter_wd * filter_ht * in_ch;
    /* Align to 16 for SIMD: zero-padded tail doesn't affect dot product */
    const int32_t window_len_aligned = (window_len + 15) & ~15;
    const int8_t pad_val = (int8_t)(-input_offset);

    /* Scratch layout (16-byte aligned):
     * [filter_sum: out_ch * 4]
     * [aligned_filter: out_ch * window_len_aligned]  -- zero-padded copy
     * [im2col_buf: window_len_aligned]
     */
    int32_t *filter_sum = (int32_t *)scratch;
    int8_t *aligned_filter = (int8_t *)((uintptr_t)((int8_t *)scratch + out_ch * sizeof(int32_t) + 15) & ~15);
    int8_t *im2col_buf = (int8_t *)((uintptr_t)(aligned_filter + out_ch * window_len_aligned + 15) & ~15);

    /* Pre-compute filter_sum * input_offset AND copy filter with zero-padded tail */
    const int8_t *fptr = filter_data;
    int8_t *af_ptr = aligned_filter;
    for (int32_t oc = 0; oc < out_ch; oc++) {
        int32_t sum = 0;
        for (int32_t fi = 0; fi < window_len; fi++) {
            sum += fptr[fi];
        }
        filter_sum[oc] = sum * input_offset;
        /* Copy filter + zero-pad tail for safe SIMD reads */
        memcpy(af_ptr, fptr, window_len);
        memset(af_ptr + window_len, 0, window_len_aligned - window_len);
        fptr += window_len;
        af_ptr += window_len_aligned;
    }

    /* Zero the tail of im2col buffer once (for aligned SIMD reads) */
    memset(im2col_buf + window_len, 0, window_len_aligned - window_len);

    /* Compute safe interior region where no bounds checking needed.
     * Interior: all filter taps fall within valid input. */
    const int32_t row_bytes = filter_wd * in_ch;
    int32_t safe_y_start = (pad_ht + stride_ht - 1) / stride_ht;
    int32_t safe_y_end = (input_ht - filter_ht + pad_ht) / stride_ht + 1;
    int32_t safe_x_start = (pad_wd + stride_wd - 1) / stride_wd;
    int32_t safe_x_end = (input_wd - filter_wd + pad_wd) / stride_wd + 1;
    if (safe_y_start > out_ht) safe_y_start = out_ht;
    if (safe_y_end > out_ht) safe_y_end = out_ht;
    if (safe_y_end < safe_y_start) safe_y_end = safe_y_start;
    if (safe_x_start > out_wd) safe_x_start = out_wd;
    if (safe_x_end > out_wd) safe_x_end = out_wd;
    if (safe_x_end < safe_x_start) safe_x_end = safe_x_start;

    /* Process each output pixel */
    int8_t *out_ptr = out_data;
    for (int32_t out_y = 0; out_y < out_ht; out_y++) {
        const int32_t base_y = out_y * stride_ht - pad_ht;
        int is_safe_y = (out_y >= safe_y_start && out_y < safe_y_end);

        for (int32_t out_x = 0; out_x < out_wd; out_x++) {
            const int32_t base_x = out_x * stride_wd - pad_wd;

            /* Copy input window into contiguous im2col buffer */
            int8_t *buf = im2col_buf;

            if (is_safe_y && out_x >= safe_x_start && out_x < safe_x_end) {
                /* FAST PATH: interior pixel — no bounds checking needed.
                 * All filter taps guaranteed to be within valid input. */
                for (int32_t fy = 0; fy < filter_ht; fy++) {
                    const int8_t *src = input_data + ((base_y + fy) * input_wd + base_x) * in_ch;
                    memcpy(buf, src, row_bytes);
                    buf += row_bytes;
                }
            } else {
                /* SLOW PATH: edge pixel — per-element bounds checking */
                for (int32_t fy = 0; fy < filter_ht; fy++) {
                    int32_t in_y = base_y + fy;
                    if (in_y >= 0 && in_y < input_ht) {
                        for (int32_t fx = 0; fx < filter_wd; fx++) {
                            int32_t in_x = base_x + fx;
                            if (in_x >= 0 && in_x < input_wd) {
                                const int8_t *src = input_data + (in_y * input_wd + in_x) * in_ch;
                                memcpy(buf, src, in_ch);
                            } else {
                                memset(buf, pad_val, in_ch);
                            }
                            buf += in_ch;
                        }
                    } else {
                        memset(buf, pad_val, row_bytes);
                        buf += row_bytes;
                    }
                }
            }

            /* Dot product against each output channel's filter (aligned copy) */
            const int32_t *out_mult_ptr = quant_data->mult;
            const int32_t *out_shift_ptr = quant_data->shift;
            const int8_t *filter_ptr = aligned_filter;

            for (int32_t oc = 0; oc < out_ch; oc++) {
                int32_t conv_out = esp_nn_dot_s8_aligned_esp32s3(im2col_buf, filter_ptr, window_len_aligned);
                conv_out += filter_sum[oc];
                if (bias) conv_out += bias[oc];
                conv_out = esp_nn_multiply_by_quantized_mult_fast(conv_out, *out_mult_ptr++, *out_shift_ptr++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_ptr++ = (int8_t) conv_out;
                filter_ptr += window_len_aligned;
            }
        }
    }
}

int esp_nn_get_conv_scratch_size_esp32s3(const data_dims_t *input_dims,
                                         const data_dims_t *filter_dims,
                                         const data_dims_t *output_dims,
                                         const conv_params_t *conv_params)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;

    int new_channels = (in_ch + 7) & ~7;

    int input_scratch = input_wd * input_ht * in_ch;
    int filter_scratch = filter_wd * filter_ht * in_ch * out_ch;

    int align_buf_size = 32; /* extra buffer for alignment */
    if ((filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0) &&
            (stride_wd == 1 && stride_ht == 1)) {
        int transpose_buf_size = 2 * (8 * new_channels); /* to store intermediate data */
        if (input_wd * input_ht < 8) {
            transpose_buf_size = 0; // not using this for leftover
        }
        if (in_ch % 8) {
            input_scratch = input_wd * input_ht * new_channels;
        } else {
            input_scratch = 0;
        }
        filter_scratch = new_channels * out_ch;
        return input_scratch + filter_scratch + transpose_buf_size + align_buf_size;
    } else {
        int32_t filter_row_size = filter_wd * in_ch;
        int32_t window_len = filter_wd * filter_ht * in_ch;

        /* Im2col path: filter_wd * in_ch < 16 but window_len >= 16 */
        if (filter_row_size < 16 && window_len >= 16) {
            int32_t window_len_aligned = (window_len + 15) & ~15;
            /* filter_sum + aligned_filter_copy + im2col_buf + alignment padding */
            int im2col_scratch = out_ch * 4 + 16 + out_ch * window_len_aligned + 16 + window_len_aligned;
            return im2col_scratch + align_buf_size;
        }

        new_channels = (in_ch + 15) & ~15;
        if (pad_wd == 0 && pad_ht == 0) {
            input_scratch = 0;
        } else {
            input_scratch = (input_wd + 2 * pad_wd) * (input_ht + 2 * pad_ht) * in_ch;
        }
        filter_scratch = filter_wd * filter_ht * new_channels * out_ch;

        // Account for filter alignment padding (worst case)
        int32_t aligned_filter_row_size = ((filter_row_size + 15) / 16) * 16;
        int filter_alignment_scratch = aligned_filter_row_size * filter_ht * out_ch;

        // Account for right/bottom padding even when pad_wd=0, pad_ht=0
        int pad_right = max(0, (output_dims->width * stride_wd + filter_wd - 1) - input_wd);
        int pad_bottom = max(0, (output_dims->height * stride_ht + filter_ht - 1) - input_ht);
        int boundary_padding_scratch = 0;
        if (pad_right > 0 || pad_bottom > 0) {
            boundary_padding_scratch = (input_wd + pad_right) * (input_ht + pad_bottom) * in_ch;
        }

        int offset_acc_scratch = out_ch * 4;
        return input_scratch + filter_scratch + filter_alignment_scratch + boundary_padding_scratch + align_buf_size + offset_acc_scratch;
    }
    return align_buf_size;
}

void esp_nn_set_conv_scratch_buf_esp32s3(void *buf)
{
    scratch_buffer = (int16_t *) buf;
}

void esp_nn_conv_s8_esp32s3(const data_dims_t *input_dims,
                            const int8_t *input,
                            const data_dims_t *filter_dims,
                            const int8_t *filter_data,
                            const int32_t *bias,
                            const data_dims_t *output_dims,
                            int8_t *out_data,
                            const conv_params_t *conv_params,
                            const quant_data_t *quant_data)
{
    if (scratch_buffer == NULL) {
        printf("esp_nn_conv error! scratch_buffer not set!\n");
        return;
    }
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_channels = output_dims->channels;
    const int32_t *out_shift = quant_data->shift;
    const int32_t *out_mult = quant_data->mult;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    int filter_size = filter_wd * filter_ht * channels * out_channels;

    if (filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0 &&
            stride_wd == 1 && stride_ht == 1) {

        int8_t *input_aligned = (int8_t *) input;
        int8_t *scratch_buf = (int8_t *) scratch_buffer;
        int8_t *filter_aligned = (int8_t *) scratch_buffer;
        int new_channels = channels;
        if (channels % 8 == 0) {
            if ((int) filter_data & 7) { // if the filter_data is not aligned to 8 bytes
                int scratch_offset = (int) (filter_aligned + filter_size);
                scratch_buf = (int8_t *) (scratch_offset + 16 - (scratch_offset & 15));
                memcpy(filter_aligned, filter_data, filter_size); // copy to aligned address
            } else {
                filter_aligned = (int8_t *) filter_data;
            }
        } else {
            // pad extra channel to make it multiple of 8. Both input and filter
            new_channels = (channels + 7) & ~7;
            for (int out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
                memcpy(filter_aligned, filter_data, channels);
                memset(filter_aligned + channels, 0, new_channels - channels);
                filter_aligned += new_channels;
                filter_data += channels;
            }
            filter_aligned = (int8_t *) scratch_buffer;
            int filter_data_size = new_channels * out_channels;
            input_aligned = filter_aligned + filter_data_size;
            for (int input_idx = 0; input_idx < input_ht * input_wd; input_idx++) {
                memcpy(input_aligned, input, channels);
                memset(input_aligned + channels, 0, new_channels - channels);
                input_aligned += new_channels;
                input += channels;
            }
            input_aligned = filter_aligned + filter_data_size;
            scratch_buf = input_aligned +  input_ht * input_wd * new_channels;
        }
        esp_nn_conv_s8_mult8_1x1_esp32s3(
            input_aligned, input_wd, input_ht, new_channels, input_offset,
            filter_aligned, bias, out_data, out_wd, out_ht, out_channels, out_offset,
            out_shift, out_mult, activation_min, activation_max, scratch_buf);
    } else {
        int32_t filter_row_size = filter_wd * channels;
        int32_t window_len = filter_wd * filter_ht * channels;

        /* Im2col path: small in_ch where per-row SIMD is wasteful,
         * but entire window is large enough for SIMD dot product.
         * E.g., 3x3 conv with in_ch=3: row=9 (<16), window=27 (>=16). */
        if (filter_row_size < 16 && window_len >= 16) {
            esp_nn_conv_s8_im2col_s3(input_dims, input, filter_dims, filter_data,
                                      bias, output_dims, out_data, conv_params,
                                      quant_data, scratch_buffer);
            return;
        }

        // align the `filter width * channels` to 16 bytes. Do zero padding for the same
        int32_t filter_alignment_padding = 16 - (filter_row_size & 15);
        int8_t *filter_data_aligned = (int8_t *) filter_data;
        int8_t *input_padded = (int8_t *) input;
        int8_t *scratch_data = (int8_t *) scratch_buffer;
        int new_input_wd = input_wd, new_input_ht = input_ht;
        if (filter_alignment_padding != 16) {
            // pad filter_data
            int32_t new_row_size = filter_wd * channels + filter_alignment_padding;
            filter_data_aligned = scratch_data;
            int8_t *row_ptr = filter_data_aligned;
            const int8_t *filter_data_ptr = filter_data;
            for (int32_t ch_idx = 0; ch_idx < out_channels; ch_idx++) {
                for (int32_t row_idx = 0; row_idx < filter_ht; row_idx++) {
                    memcpy(row_ptr, filter_data_ptr, filter_row_size);
                    memset(row_ptr + filter_row_size, 0, new_row_size - filter_row_size);
                    filter_data_ptr += filter_row_size;
                    row_ptr += new_row_size;
                }
            }
            scratch_data += new_row_size * filter_ht * out_channels;
            filter_row_size = new_row_size;
        } else if ((int) filter_data & 15) {
            filter_data_aligned = scratch_data;
            memcpy(filter_data_aligned, filter_data, filter_size);
            scratch_data += filter_size;
        }
        // Calculate if right/bottom padding is needed even when pad_wd=0, pad_ht=0
        // This happens when the filter extends beyond input boundaries at the edges
        // Formula matches depthwise convolution: (out_wd * stride_wd + filter_wd - 1) - input_wd
        int32_t pad_right = max(0, (out_wd * stride_wd + filter_wd - 1) - input_wd);
        int32_t pad_bottom = max(0, (out_ht * stride_ht + filter_ht - 1) - input_ht);

        // Apply padding if explicitly requested (pad_wd/pad_ht) OR if needed for boundary handling
        if (pad_wd != 0 || pad_ht != 0) {
            // Full padding (top, bottom, left, right) when pad_wd/pad_ht are set
            input_padded = (int8_t *) scratch_data;
            esp_nn_aligned_s8_pad_with_value(input, input_padded, input_wd, input_ht, channels,
                                            -input_offset, pad_wd, pad_ht);
            new_input_wd = input_wd + 2 * pad_wd;
            new_input_ht = input_ht + 2 * pad_ht;
            scratch_data += new_input_wd * new_input_ht * channels;
        } else if (pad_right > 0 || pad_bottom > 0) {
            // Only right/bottom padding needed for boundary handling (like depthwise conv)
            input_padded = (int8_t *) scratch_data;
            esp_nn_aligned_s8_pad_end_with_value(input, input_padded, input_wd, input_ht, channels,
                                                -input_offset, (uint16_t)pad_right, (uint16_t)pad_bottom);
            new_input_wd = input_wd + pad_right;
            new_input_ht = input_ht + pad_bottom;
            scratch_data += new_input_wd * new_input_ht * channels;
        }

        // Pre-compute per-channel offset corrections in C wrapper for large filters.
        // This avoids the assembly scanning all filter data (DCache pollution).
        // Only do this for large filters where the benefit outweighs overhead.
        int filter_total = filter_wd * filter_ht * channels * out_channels;
        if (input_offset != 0 && filter_total > 16384) {
            int32_t *corrections = (int32_t *)scratch_data;
            int32_t filter_ch_size = filter_wd * filter_ht * channels;
            const int8_t *f_src = filter_data; // use ORIGINAL (not aligned) filter for sum
            for (int ch = 0; ch < out_channels; ch++) {
                int32_t filter_sum = 0;
                for (int i = 0; i < filter_ch_size; i++) {
                    filter_sum += f_src[i];
                }
                corrections[ch] = filter_sum * input_offset;
                if (bias) {
                    corrections[ch] += bias[ch];
                }
                f_src += filter_ch_size;
            }
            // Pass input_offset=0 to assembly so it skips its pre-computation.
            // Pass scratch_data as "bias" pointer — the assembly's bias-copy loop
            // will read from scratch and write to scratch (identity, no-op).
            esp_nn_conv_s8_filter_aligned_input_padded_esp32s3(
                input_padded, new_input_wd, new_input_ht, channels, 0,
                stride_wd, stride_ht, filter_data_aligned, filter_wd, filter_ht,
                (const int32_t *)scratch_data, out_data, out_wd, out_ht, out_channels,
                out_offset, out_shift, out_mult, activation_min, activation_max,
                scratch_data);
        } else {
            esp_nn_conv_s8_filter_aligned_input_padded_esp32s3(
                input_padded, new_input_wd, new_input_ht, channels, input_offset,
                stride_wd, stride_ht, filter_data_aligned, filter_wd, filter_ht,
                bias, out_data, out_wd, out_ht, out_channels, out_offset,
                out_shift, out_mult, activation_min, activation_max, scratch_data);
        }
    }
}
