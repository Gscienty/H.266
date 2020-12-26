/*
 * Copyright (c) 2020 Gscienty <gaoxiaochuan@hotmail.com>
 *
 * Distributed under the MIT software license, see the accompan ying
 * file LICENSE or https://www.opensource.org/licenses/mit-lice nse.php .
 *
 */

#ifndef __H_266_RGB_H__
#define __H_266_RGB_H__

#include <stdint.h>

typedef struct vvc_rgb_s vvc_rgb_t;
struct vvc_rgb_s {
    uint8_t blue;
    uint8_t green;
    uint8_t red;
};

typedef struct vvc_rgb_graph_s vvc_rgb_graph_t;
struct vvc_rgb_graph_s {
    uint32_t width;
    uint32_t height;

    vvc_rgb_t payload[0];
};

#endif
