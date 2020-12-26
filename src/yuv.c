/*
 * Copyright (c) 2020 Gscienty <gaoxiaochuan@hotmail.com>
 *
 * Distributed under the MIT software license, see the accompan ying
 * file LICENSE or https://www.opensource.org/licenses/mit-lice nse.php .
 *
 */

#include "yuv.h"
#include <tmmintrin.h>
#include <string.h>

#include <stdio.h>

static int vvc_rgb2yuv_yuyv(uint8_t *const out, uint32_t outlen, vvc_rgb_graph_t *const rgb_graph);
static int vvc_yuv2rgb_yuyv(vvc_rgb_graph_t *const rgb_graph, uint8_t *const in);

typedef enum vvc_sse_calcgroup_type_e vvc_sse_calcgroup_type_t;
enum vvc_sse_calcgroup_type_e {
    vvc_sse_calcgroup_type_y = 0,
    vvc_sse_calcgroup_type_u,
    vvc_sse_calcgroup_type_v,

    vvc_sse_calcgroup_type_b = 3,
    vvc_sse_calcgroup_type_g,
    vvc_sse_calcgroup_type_r,
};

typedef uint8_t vvc_sse_calcgroup_t[48];

static __attribute((always_inline)) inline __m128i vvc_rgb2yuv_calc(__m128i sse_src1, __m128i sse_src2, __m128i sse_src3, vvc_sse_calcgroup_type_t calctype) {
    __m128i sse_bgw;
    __m128i sse_crw;
    const __m128i sse_cmask = _mm_setr_epi16(128, 0, 128, 0, 128, 0, 128, 0);

    const int sse_rshift = 13;

    // 16bits:
    // B0 00 G0 00 B1 00 G1 00 B2 00 G2 00 B3 00 G3 00
    __m128i sse_llbg = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(0, -1, 1, -1, 3, -1, 4, -1, 6, -1, 7, -1, 9, -1, 10, -1));
    // B4 00 G4 00 B5 00 00 00 00 00 00 00 00 00 00 00
    __m128i sse_lhbg = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(12, -1, 13, -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    // 00 00 00 00 00 00 G5 00 B6 00 G6 00 B7 00 G7 00
    sse_lhbg = _mm_or_si128(sse_lhbg, _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, -1, 2, -1, 3, -1, 5, -1, 6, -1)));
    // B8 00 G8 00 B9 00 G9 00 BA 00 GA 00 00 00 00 00
    __m128i sse_hlbg = _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(8, -1, 9, -1, 11, -1, 12, -1, 14, -1, 15, -1, -1, -1, -1, -1));
    // 00 00 00 00 00 00 00 00 00 00 00 00 BB 00 GB 00
    sse_hlbg = _mm_or_si128(sse_hlbg, _mm_shuffle_epi8(sse_src3, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1)));
    // BC 00 GC 00 BD 00 GD 00 BE 00 GE 00 BF 00 GF 00
    __m128i sse_hhbg = _mm_shuffle_epi8(sse_src3, _mm_setr_epi8(4, -1, 5, -1, 7, -1, 8, -1, 10, -1, 11, -1, 13, -1, 14, -1));

    // 00 00 R0 00 00 00 R1 00 00 00 R2 00 00 00 R3 00
    __m128i sse_llcr = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, -1, 2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1));
    // 00 00 R4 00 00 00 00 00 00 00 00 00 00 00 00 00
    __m128i sse_lhcr = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    // 00 00 00 00 00 00 R5 00 00 00 R6 00 00 00 R7 00
    sse_lhcr = _mm_or_si128(sse_lhcr, _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 4, -1, -1, -1, 7, -1)));
    // 00 00 R8 00 00 00 R9 00 00 00 00 00 00 00 00 00
    __m128i sse_hlcr = _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(-1, -1, 10, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    // 00 00 00 00 00 00 00 00 00 00 RA 00 00 00 RB 00
    sse_hlcr = _mm_or_si128(sse_lhcr, _mm_shuffle_epi8(sse_src3, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 3, -1)));
    // 00 00 RC 00 00 00 RD 00 00 00 RE 00 00 00 RF 00
    __m128i sse_hhcr = _mm_shuffle_epi8(sse_src3, _mm_setr_epi8(-1, -1, 6, -1, -1, -1, 9, -1, -1, -1, 12, -1, -1, -1, 15, -1));

    sse_llcr = _mm_or_si128(sse_llcr, sse_cmask);
    sse_lhcr = _mm_or_si128(sse_lhcr, sse_cmask);
    sse_hlcr = _mm_or_si128(sse_hlcr, sse_cmask);
    sse_hhcr = _mm_or_si128(sse_hhcr, sse_cmask);

    switch (calctype) {
    case vvc_sse_calcgroup_type_y:
        sse_bgw = _mm_setr_epi16(934, 4808, 934, 4808, 934, 4808, 934, 4808);
        sse_crw = _mm_setr_epi16(0, 2449, 0, 2449, 0, 2449, 0, 2449);
        break;

    case vvc_sse_calcgroup_type_u:
        sse_bgw = _mm_setr_epi16(4096, -2711, 4096, -2711, 4096, -2711, 4096, -2711);
        sse_crw = _mm_setr_epi16(8192, -1384, 8192, -1384, 8192, -1384, 8192, -1384);
        break;

    case vvc_sse_calcgroup_type_v:
        sse_bgw = _mm_setr_epi16(-663, -3432, -663, -3432, -663, -3432, -663, -3432);
        sse_crw = _mm_setr_epi16(8192, 4096, 8192, 4096, 8192, 4096, 8192, 4096);
        break;

    default:
        return _mm_setr_epi32(0, 0, 0, 0);
    }

    __m128i sse_ll = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_llbg, sse_bgw), _mm_madd_epi16(sse_llcr, sse_crw)), sse_rshift);
    __m128i sse_lh = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_lhbg, sse_bgw), _mm_madd_epi16(sse_lhcr, sse_crw)), sse_rshift);
    __m128i sse_hl = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_hlbg, sse_bgw), _mm_madd_epi16(sse_hlcr, sse_crw)), sse_rshift);
    __m128i sse_hh = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_hhbg, sse_bgw), _mm_madd_epi16(sse_hhcr, sse_crw)), sse_rshift);

    __m128i sse_l = _mm_packs_epi32(sse_ll, sse_lh);
    __m128i sse_h = _mm_packs_epi32(sse_hl, sse_hh);

    return _mm_packus_epi16(sse_l, sse_h);
}

static __attribute((always_inline)) inline __m128i vvc_yuyv2rgb(__m128i sse_src1, __m128i sse_src2, vvc_sse_calcgroup_type_t calctype) {
    __m128i sse_yuw;
    __m128i sse_cvw;
    const __m128i sse_csub = _mm_setr_epi16(128, 0, 128, 0, 128, 0, 128, 0);

    const int sse_rshift = 13;

    // Y0 00 00 00 Y1 00 00 00 Y2 00 00 00 Y3 00 00 00
    __m128i sse_llyu = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(0, -1, -1, -1, 2, -1, -1, -1, 4, -1, -1, -1, 6, -1, -1, -1));
    // 00 00 U0 00 00 00 U0 00 00 00 U1 00 00 00 U1 00
    sse_llyu = _mm_or_si128(sse_llyu, _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 5, -1, -1, -1, 5, -1)));
    // Y4 00 00 00 Y5 00 00 00 Y6 00 00 00 Y7 00 00 00
    __m128i sse_lhyu = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(8, -1, -1, -1, 10, -1, -1, -1, 12, -1, -1, -1, 14, -1, -1, -1));
    // 00 00 U2 00 00 00 U2 00 00 00 U3 00 00 00 U3 00
    sse_lhyu = _mm_or_si128(sse_lhyu, _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, -1, 9, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1, 13, -1)));
    // Y8 00 00 00 Y9 00 00 00 YA 00 00 00 YB 00 00 00
    __m128i sse_hlyu = _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(0, -1, -1, -1, 2, -1, -1, -1, 4, -1, -1, -1, 6, -1, -1, -1));
    // 00 00 U4 00 00 00 U4 00 00 00 U5 00 00 00 U5 00
    sse_hlyu = _mm_or_si128(sse_hlyu, _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(-1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 5, -1, -1, -1, 5, -1)));
    // YC 00 00 00 YD 00 00 00 YE 00 00 00 YF 00 00 00
    __m128i sse_hhyu = _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(8, -1, -1, -1, 10, -1, -1, -1, 12, -1, -1, -1, 14, -1, -1, -1));
    // 00 00 U6 00 00 00 U6 00 00 00 U7 00 00 00 U7 00
    sse_hhyu = _mm_or_si128(sse_hhyu, _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(-1, -1, 9, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1, 13, -1)));

    // 00 00 V0 00 00 00 V0 00 00 00 V1 00 00 00 V1 00
    __m128i sse_llcv = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, -1, 3, -1, -1, -1, 3, -1, -1, -1, 7, -1, -1, -1, 7, -1));
    // 00 00 V2 00 00 00 V2 00 00 00 V3 00 00 00 V3 00
    __m128i sse_lhcv = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, -1, 11, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1, 15, -1));
    // 00 00 V4 00 00 00 V4 00 00 00 V5 00 00 00 V5 00
    __m128i sse_hlcv = _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(-1, -1, 3, -1, -1, -1, 3, -1, -1, -1, 7, -1, -1, -1, 7, -1));
    // 00 00 V6 00 00 00 V6 00 00 00 V7 00 00 00 V7 00
    __m128i sse_hhcv = _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(-1, -1, 11, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1, 15, -1));

    sse_llcv = _mm_or_si128(sse_llcv, sse_csub);
    sse_lhcv = _mm_or_si128(sse_lhcv, sse_csub);
    sse_hlcv = _mm_or_si128(sse_hlcv, sse_csub);
    sse_hhcv = _mm_or_si128(sse_hhcv, sse_csub);

    switch (calctype) {
    case vvc_sse_calcgroup_type_b:
        sse_yuw = _mm_setr_epi16(8192, 16647, 8192, 16647, 8192, 16647, 8192, 16647);
        sse_cvw = _mm_setr_epi16(-16647, 0, -16647, 0, -16647, 0, -16647, 0);
        break;

    case vvc_sse_calcgroup_type_g:
        sse_yuw = _mm_setr_epi16(8192, -3233, 8192, -3233, 8192, -3233, 8192, -3233);
        sse_cvw = _mm_setr_epi16(7989, -4756, 7989, -4756, 7989, -4756, 7989, -4756);
        break;

    case vvc_sse_calcgroup_type_r:
        sse_yuw = _mm_setr_epi16(8192, 0, 8192, 0, 8192, 0, 8192, 0);
        sse_cvw = _mm_setr_epi16(-9337, 9337, -9337, 9337, -9337, 9337, -9337, 9337);
        break;

    default:
        return _mm_setr_epi32(0, 0, 0, 0);
    }

    __m128i sse_ll = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_llyu, sse_yuw), _mm_madd_epi16(sse_llcv, sse_cvw)), sse_rshift);
    __m128i sse_lh = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_lhyu, sse_yuw), _mm_madd_epi16(sse_lhcv, sse_cvw)), sse_rshift);
    __m128i sse_hl = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_hlyu, sse_yuw), _mm_madd_epi16(sse_hlcv, sse_cvw)), sse_rshift);
    __m128i sse_hh = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_hhyu, sse_yuw), _mm_madd_epi16(sse_hhcv, sse_cvw)), sse_rshift);

    __m128i sse_l = _mm_packs_epi32(sse_ll, sse_lh);
    __m128i sse_h = _mm_packs_epi32(sse_hl, sse_hh);

    return _mm_packus_epi16(sse_l, sse_h);
}

static int vvc_rgb2yuv_yuyv(uint8_t *const out, uint32_t outlen, vvc_rgb_graph_t *const rgb_graph) {
    uint32_t row;
    uint32_t col;
    if (outlen != vvc_yuv_spschroma_formatted_size(rgb_graph->width, rgb_graph->height, vvc_yuv_spschroma_format_422)) {
        return -1;
    }
    uint32_t block = rgb_graph->width / 16;
    for (row = 0; row < rgb_graph->height; row++) {
        for (col = 0; col < block * 16; col += 16) {
            __m128i *off = (__m128i *) &rgb_graph->payload[row * rgb_graph->width + col];

            // B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3 B4 G4 R4 B5
            __m128i sse_src1 = _mm_loadu_si128(off);
            // G5 R5 B6 G6 R6 B7 G7 R7 B8 G8 R8 B9 G9 R9 BA GA
            __m128i sse_src2 = _mm_loadu_si128(off + 1);
            // RA BB GB RB BC GC RC BD GD RD BE GE RE BF GF RF
            __m128i sse_src3 = _mm_loadu_si128(off + 2);

            __m128i calcy = vvc_rgb2yuv_calc(sse_src1, sse_src2, sse_src3, vvc_sse_calcgroup_type_y);
            __m128i calcu = vvc_rgb2yuv_calc(sse_src1, sse_src2, sse_src3, vvc_sse_calcgroup_type_u);
            __m128i calcv = vvc_rgb2yuv_calc(sse_src1, sse_src2, sse_src3, vvc_sse_calcgroup_type_v);

            uint8_t *y = (uint8_t *) &calcy;
            uint8_t *u = (uint8_t *) &calcu;
            uint8_t *v = (uint8_t *) &calcv;

            vvc_sse_calcgroup_t yuyv = {
                y[0],  u[0],  y[1],  v[0],
                y[2],  u[2],  y[3],  v[2],
                y[4],  u[4],  y[5],  v[4],
                y[6],  u[6],  y[7],  v[6],
                y[8],  u[8],  y[9],  v[8],
                y[10], u[10], y[11], v[10],
                y[12], u[12], y[13], v[12],
                y[14], u[14], y[15], v[14],
            };

            memcpy(&out[row * rgb_graph->width + col], &yuyv, sizeof(yuyv));
        }
    }
    return 0;
}

static int vvc_yuv2rgb_yuyv(vvc_rgb_graph_t *const rgb_graph, uint8_t *const in) {
    const size_t in_len = vvc_yuv_spschroma_formatted_size(rgb_graph->width, rgb_graph->height, vvc_yuv_spschroma_format_422);
    size_t i;
    for (i = 0; i < in_len; i += 32) {
        __m128i *off = (__m128i *) &in[i];

        // Y0 U0 Y1 V0 Y2 U1 Y3 V1 Y4 U2 Y5 V2 Y6 U3 Y7 V3
        __m128i sse_src1 = _mm_loadu_si128(off);
        // Y8 U4 Y9 V4 YA U5 YB V5 YC U6 YD V6 YE U7 YF V7
        __m128i sse_src2 = _mm_loadu_si128(off + 1);

        __m128i calcb = vvc_yuyv2rgb(sse_src1, sse_src2, vvc_sse_calcgroup_type_b);
        __m128i calcg = vvc_yuyv2rgb(sse_src1, sse_src2, vvc_sse_calcgroup_type_g);
        __m128i calcr = vvc_yuyv2rgb(sse_src1, sse_src2, vvc_sse_calcgroup_type_r);

        uint8_t *b = (uint8_t *) &calcb;
        uint8_t *g = (uint8_t *) &calcg;
        uint8_t *r = (uint8_t *) &calcr;

        vvc_rgb_t rgb[16] = {
            { .blue = b[0], .green = g[0], .red = r[0] },
            { .blue = b[1], .green = g[1], .red = r[1] },
            { .blue = b[2], .green = g[2], .red = r[2] },
            { .blue = b[3], .green = g[3], .red = r[3] },
            { .blue = b[4], .green = g[4], .red = r[4] },
            { .blue = b[5], .green = g[5], .red = r[5] },
            { .blue = b[6], .green = g[6], .red = r[6] },
            { .blue = b[7], .green = g[7], .red = r[7] },
            { .blue = b[8], .green = g[8], .red = r[8] },
            { .blue = b[9], .green = g[9], .red = r[9] },
            { .blue = b[10], .green = g[10], .red = r[10] },
            { .blue = b[11], .green = g[11], .red = r[11] },
            { .blue = b[12], .green = g[12], .red = r[12] },
            { .blue = b[13], .green = g[13], .red = r[13] },
            { .blue = b[14], .green = g[14], .red = r[14] },
            { .blue = b[15], .green = g[15], .red = r[15] },
        };

        memcpy(&rgb_graph->payload[i << 1], &rgb, sizeof(rgb));
    }
    return 0;
}

int vvc_rgb2yuv(uint8_t *const out, uint32_t outlen, vvc_rgb_graph_t *const rgb_graph, const vvc_yuv_format_t format) {
    switch (format) {

    case vvc_yuv_format_yuyv:
        return vvc_rgb2yuv_yuyv(out, outlen, rgb_graph);
    default:
        return -2;
    }
}

int vvc_yuv2rgb(vvc_rgb_graph_t *const rgb_graph, uint8_t *const in, const vvc_yuv_format_t format) {
    switch (format) {
    case vvc_yuv_format_yuyv:
        return vvc_yuv2rgb_yuyv(rgb_graph, in);

    default:
        return -2;
    }
}
