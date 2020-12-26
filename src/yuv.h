/*
 * Copyright (c) 2020 Gscienty <gaoxiaochuan@hotmail.com>
 *
 * Distributed under the MIT software license, see the accompan ying
 * file LICENSE or https://www.opensource.org/licenses/mit-lice nse.php .
 *
 */

#ifndef __H_266_YUV_H__
#define __H_266_YUV_H__

#include "rgb.h"
#include <stdint.h>

/*
 * RGB2YUV
 * 
 * R = Y + 1.13983 * (V - 128)
 * G = Y - 0.39465 * (U - 128) - 0.58060 * (V - 128)
 * B = Y + 2.03211 * (U - 128)
 *
 * Y = 0.299 * R + 0.587 * G + 0.114 * B
 * U = -0.169 * R - 0.331 * G + 0.5 * B + 128
 * V = 0.5 * R - 0.419 * G - 0.081 * B + 128
 */

typedef enum vvc_yuv_spschroma_format_e vvc_yuv_spschroma_format_t;
enum vvc_yuv_spschroma_format_e {
    vvc_yuv_spschroma_format_monochrome = 0,
    vvc_yuv_spschroma_format_420,
    vvc_yuv_spschroma_format_422,
    vvc_yuv_spschroma_format_444,
};

typedef enum vvc_yuv_format_e vvc_yuv_format_t;
enum vvc_yuv_format_e {
    vvc_yuv_format_yuyv = 0,
    vvc_yuv_format_uyvy,
    vvc_yuv_format_yuv422p,
    vvc_yuv_format_yu12,
    vvc_yuv_format_yu21,
    vvc_yuv_format_nv12,
    vvc_yuv_format_nv21,
};

#define vvc_yuv_spschroma_formatted_monochroma_size(w, h) ((w) * (h))
#define vvc_yuv_spschroma_formatted_420_size(w, h) (((w) * (h) * 3) >> 1)
#define vvc_yuv_spschroma_formatted_422_size(w, h) (((w) * (h) * 2))
#define vvc_yuv_spschroma_formatted_444_size(w, h) (((w) * (h) * 3))

static inline uint32_t vvc_yuv_spschroma_formatted_size(const uint32_t w, const uint32_t h, vvc_yuv_spschroma_format_t format) {
    switch (format) {
    case vvc_yuv_spschroma_format_monochrome:
        return vvc_yuv_spschroma_formatted_monochroma_size(w, h);
    case vvc_yuv_spschroma_format_420:
        return vvc_yuv_spschroma_formatted_420_size(w, h);
    case vvc_yuv_spschroma_format_422:
        return vvc_yuv_spschroma_formatted_422_size(w, h);
    case vvc_yuv_spschroma_format_444:
        return vvc_yuv_spschroma_formatted_444_size(w, h);
    }
}

#define vvc_rgb2y(rgb) \
    (((uint32_t) (rgb).red * 299 + (uint32_t) (rgb).green * 587 + (uint32_t) (rgb).blue * 114) / 1000)

#define vvc_rgb2u(rgb) \
    ((128000 - (uint32_t) (rgb).red * 169 - (uint32_t) (rgb).green * 331 + (uint32_t) (rgb).blue * 500) / 1000)

#define vvc_rgb2v(rgb) \
    ((128000 + (uint32_t) (rgb).red * 500 - (uint32_t) (rgb).green * 419 - (uint32_t) (rgb).blue * 81) / 1000)

int vvc_rgb2yuv(uint8_t *const out, uint32_t outlen, vvc_rgb_graph_t *const rgb_graph, const vvc_yuv_format_t format);

int vvc_yuv2rgb(vvc_rgb_graph_t *const rgb_graph, uint8_t *const in, const vvc_yuv_format_t format);

#endif
