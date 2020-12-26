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
static int vvc_rgb2yuv_uyuv(uint8_t *const out, uint32_t outlen, vvc_rgb_graph_t *const rgb_graph);

typedef enum vvc_sse_calcgroup_type_e vvc_sse_calcgroup_type_t;
enum vvc_sse_calcgroup_type_e {
    vvc_sse_calcgroup_type_y = 0,
    vvc_sse_calcgroup_type_u,
    vvc_sse_calcgroup_type_v,
};

typedef uint8_t vvc_sse_calcgroup_t[48];

static __attribute((always_inline)) inline __m128i vvc_rgb2yuv_calc(__m128i sse_src1, __m128i sse_src2, __m128i sse_src3, vvc_sse_calcgroup_type_t calctype) {
    __m128i sse_ybgw;
    __m128i sse_ycrw;
    const __m128i sse_cmask = _mm_setr_epi16(128, 0, 128, 0, 128, 0, 128, 0);

    const int sse_rshift = 13;

    // 16bits:
    // 00 B0 00 G0 00 B1 00 G1 00 B2 00 G2 00 B3 00 G3
    __m128i sse_llbg = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, 0, -1, 1, -1, 3, -1, 4, -1, 6, -1, 7, -1, 9, -1, 10));
    // 00 B4 00 G4 00 B5 00 00 00 00 00 00 00 00 00 00
    __m128i sse_lhbg = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, 12, -1, 13, -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    // 00 00 00 00 00 00 00 G5 00 B6 00 G6 00 B7 00 G7 
    sse_lhbg = _mm_or_si128(sse_lhbg, _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, 0, -1, 2, -1, 3, -1, 5, -1, 6)));
    // 00 B8 00 G8 00 B9 00 G9 00 BA 00 GA 00 00 00 00
    __m128i sse_hlbg = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, 8, -1, 9, -1, 11, -1, 12, -1, 14, -1, 15, -1, -1, -1, -1));
    // 00 00 00 00 00 00 00 00 00 00 00 00 00 BB 00 GB
    sse_hlbg = _mm_or_si128(sse_hlbg, _mm_shuffle_epi8(sse_src3, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2)));
    // 00 BC 00 GC 00 BD 00 GD 00 BE 00 GE 00 BF 00 GF
    __m128i sse_hhbg = _mm_shuffle_epi8(sse_src3, _mm_setr_epi8(-1, 4, -1, 5, -1, 7, -1, 8, -1, 10, -1, 11, -1, 13, -1, 14));

    // 00 00 00 R0 00 00 00 R1 00 00 00 R2 00 00 00 R3
    __m128i sse_llcr = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, -1, -1, 2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11));
    // 00 00 00 R4 00 00 00 00 00 00 00 00 00 00 00 00
    __m128i sse_lhcr = _mm_shuffle_epi8(sse_src1, _mm_setr_epi8(-1, -1, -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    // 00 00 00 00 00 00 00 R5 00 00 00 R6 00 00 00 R7
    sse_lhcr = _mm_or_si128(sse_lhcr, _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 4, -1, -1, -1, 7)));
    // 00 00 00 R8 00 00 00 R9 00 00 00 00 00 00 00 00
    __m128i sse_hlcr = _mm_shuffle_epi8(sse_src2, _mm_setr_epi8(-1, -1, -1, 10, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1));
    // 00 00 00 00 00 00 00 00 00 00 00 RA 00 00 00 RB
    sse_hlcr = _mm_or_si128(sse_lhcr, _mm_shuffle_epi8(sse_src3, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 3)));
    // 00 00 00 RC 00 00 00 RD 00 00 00 RE 00 00 00 RF
    __m128i sse_hhcr = _mm_shuffle_epi8(sse_src3, _mm_setr_epi8(-1, -1, -1, 6, -1, -1, -1, 9, -1, -1, -1, 12, -1, -1, -1, 15));

    __m128i sse_ll;
    __m128i sse_lh;
    __m128i sse_hl;
    __m128i sse_hh;

    switch (calctype) {
    case vvc_sse_calcgroup_type_y:
        sse_ybgw = _mm_setr_epi16(933, 4808, 933, 4808, 933, 4808, 933, 4808);
        sse_ycrw = _mm_setr_epi16(0, 2449, 0, 2449, 0, 2449, 0, 2449);
        break;

    case vvc_sse_calcgroup_type_u:
        sse_ybgw = _mm_setr_epi16(4096, -2711, 4096, -2711, 4096, -2711, 4096, -2711);
        sse_ycrw = _mm_setr_epi16(8192, -1384, 8192, -1384, 8192, -1384, 8192, -1384);

        sse_llcr = _mm_or_si128(sse_llcr, sse_cmask);
        sse_lhcr = _mm_or_si128(sse_lhcr, sse_cmask);
        sse_hlcr = _mm_or_si128(sse_hlcr, sse_cmask);
        sse_hhcr = _mm_or_si128(sse_hhcr, sse_cmask);
        break;

    case vvc_sse_calcgroup_type_v:
        sse_ybgw = _mm_setr_epi16(-663, -3432, -663, -3432, -663, -3432, -663, -3432);
        sse_ycrw = _mm_setr_epi16(8192, 4096, 8192, 4096, 8192, 4096, 8192, 4096);

        sse_llcr = _mm_or_si128(sse_llcr, sse_cmask);
        sse_lhcr = _mm_or_si128(sse_lhcr, sse_cmask);
        sse_hlcr = _mm_or_si128(sse_hlcr, sse_cmask);
        sse_hhcr = _mm_or_si128(sse_hhcr, sse_cmask);
        break;
    }

    sse_ll = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_llbg, sse_ybgw), _mm_madd_epi16(sse_llcr, sse_ycrw)), sse_rshift);
    sse_lh = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_lhbg, sse_ybgw), _mm_madd_epi16(sse_lhcr, sse_ycrw)), sse_rshift);
    sse_hl = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_hlbg, sse_ybgw), _mm_madd_epi16(sse_hlcr, sse_ycrw)), sse_rshift);
    sse_hh = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(sse_hhbg, sse_ybgw), _mm_madd_epi16(sse_hhcr, sse_ycrw)), sse_rshift);

    __m128i sse_l = _mm_packs_epi32(sse_ll, sse_lh);
    __m128i sse_h = _mm_packs_epi32(sse_hl, sse_hh);

    return _mm_packs_epi16(sse_l, sse_h);
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

int vvc_rgb2yuv(uint8_t *const out, uint32_t outlen, vvc_rgb_graph_t *const rgb_graph, const vvc_yuv_format_t format) {

    switch (format) {

    case vvc_yuv_format_yuyv:
        return vvc_rgb2yuv_yuyv(out, outlen, rgb_graph);
    default:
        return -2;
    }

    return 0;
}
