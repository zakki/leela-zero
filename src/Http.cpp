/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "config.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "GTP.h"
#include "GameState.h"
#include "Http.h"
#include "NNCache.h"
#include "Network.h"
#include "Utils.h"

#include "third_party/httplib.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "third_party/stb_image_resize.h"


using namespace Utils;

#if 1
const int color_map[] = {
    0x5548C1, 0x564AC2, 0x584CC4, 0x594EC6, 0x5A4FC7, 0x5B51C9, 0x5C53CA, 0x5D55CC,
    0x5E57CD, 0x5F58CF, 0x605AD0, 0x615CD1, 0x625ED3, 0x635FD4, 0x6461D6, 0x6663D7,
    0x6765D8, 0x6866DA, 0x6968DB, 0x6A6ADC, 0x6B6BDD, 0x6C6DDF, 0x6D6FE0, 0x6E71E1,
    0x6F72E2, 0x7074E3, 0x7176E4, 0x7377E6, 0x7479E7, 0x757BE8, 0x767CE9, 0x777EEA,
    0x787FEB, 0x7981EC, 0x7A83ED, 0x7B84EE, 0x7D86EE, 0x7E87EF, 0x7F89F0, 0x808BF1,
    0x818CF2, 0x828EF3, 0x838FF3, 0x8491F4, 0x8592F5, 0x8794F6, 0x8895F6, 0x8997F7,
    0x8A98F8, 0x8B9AF8, 0x8C9BF9, 0x8D9DF9, 0x8E9EFA, 0x90A0FA, 0x91A1FB, 0x92A2FB,
    0x93A4FC, 0x94A5FC, 0x95A6FD, 0x96A8FD, 0x97A9FD, 0x99AAFE, 0x9AACFE, 0x9BADFE,
    0x9CAEFE, 0x9DB0FF, 0x9EB1FF, 0x9FB2FF, 0xA0B3FF, 0xA2B5FF, 0xA3B6FF, 0xA4B7FF,
    0xA5B8FF, 0xA6B9FF, 0xA7BAFF, 0xA8BBFF, 0xA9BDFF, 0xABBEFF, 0xACBFFF, 0xADC0FF,
    0xAEC1FF, 0xAFC2FF, 0xB0C3FF, 0xB1C4FE, 0xB2C5FE, 0xB3C6FE, 0xB4C7FD, 0xB6C8FD,
    0xB7C9FD, 0xB8C9FC, 0xB9CAFC, 0xBACBFC, 0xBBCCFB, 0xBCCDFB, 0xBDCEFA, 0xBECEFA,
    0xBFCFF9, 0xC0D0F9, 0xC1D1F8, 0xC2D1F7, 0xC3D2F7, 0xC4D3F6, 0xC5D3F5, 0xC6D4F5,
    0xC7D4F4, 0xC8D5F3, 0xC9D6F2, 0xCAD6F2, 0xCBD7F1, 0xCCD7F0, 0xCDD8EF, 0xCED8EE,
    0xCFD9ED, 0xD0D9EC, 0xD1D9EB, 0xD2DAEA, 0xD3DAE9, 0xD4DAE8, 0xD5DBE7, 0xD6DBE6,
    0xD6DBE5, 0xD7DCE4, 0xD8DCE3, 0xD9DCE2, 0xDADCE1, 0xDBDCE0, 0xDBDCDE, 0xDCDDDD,
    0xDDDCDC, 0xDEDCDB, 0xDFDCD9, 0xE0DBD8, 0xE1DBD6, 0xE2DAD5, 0xE3D9D3, 0xE3D9D2,
    0xE4D8D1, 0xE5D8CF, 0xE6D7CE, 0xE7D6CC, 0xE7D6CB, 0xE8D5C9, 0xE9D4C8, 0xE9D3C6,
    0xEAD3C5, 0xEBD2C3, 0xEBD1C2, 0xECD0C0, 0xECCFBE, 0xEDCFBD, 0xEDCEBB, 0xEECDBA,
    0xEECCB8, 0xEFCBB7, 0xEFCAB5, 0xF0C9B4, 0xF0C8B2, 0xF0C7B1, 0xF1C6AF, 0xF1C5AD,
    0xF1C4AC, 0xF2C3AA, 0xF2C2A9, 0xF2C1A7, 0xF2BFA6, 0xF3BEA4, 0xF3BDA2, 0xF3BCA1,
    0xF3BB9F, 0xF3B99E, 0xF3B89C, 0xF3B79A, 0xF3B699, 0xF3B497, 0xF3B396, 0xF3B294,
    0xF3B093, 0xF3AF91, 0xF3AE8F, 0xF3AC8E, 0xF3AB8C, 0xF3A98B, 0xF3A889, 0xF2A788,
    0xF2A586, 0xF2A485, 0xF2A283, 0xF1A181, 0xF19F80, 0xF19E7E, 0xF09C7D, 0xF09A7B,
    0xF0997A, 0xEF9778, 0xEF9677, 0xEE9475, 0xEE9274, 0xED9172, 0xED8F71, 0xEC8D6F,
    0xEC8C6E, 0xEB8A6C, 0xEB886B, 0xEA8669, 0xEA8568, 0xE98366, 0xE88165, 0xE87F63,
    0xE77E62, 0xE67C61, 0xE57A5F, 0xE5785E, 0xE4765C, 0xE3745B, 0xE27259, 0xE17058,
    0xE06F57, 0xE06D55, 0xDF6B54, 0xDE6953, 0xDD6751, 0xDC6550, 0xDB634E, 0xDA614D,
    0xD95F4C, 0xD85D4B, 0xD75B49, 0xD65848, 0xD55647, 0xD45445, 0xD25244, 0xD15043,
    0xD04E42, 0xCF4B40, 0xCE493F, 0xCD473E, 0xCB443D, 0xCA423B, 0xC9403A, 0xC83D39,
    0xC63B38, 0xC53837, 0xC43636, 0xC23334, 0xC13033, 0xC02D32, 0xBE2A31, 0xBD2730,
    0xBB242F, 0xBA212E, 0xB91D2D, 0xB7192C, 0xB6142B, 0xB40F29, 0xB30828, 0xB10127,
};
#else
const int color_map[] = {
    0x000000, 0x050004, 0x090008, 0x0D010D, 0x110111, 0x140114, 0x160117, 0x19011A,
    0x1B011E, 0x1C0221, 0x1E0225, 0x1F0228, 0x20022C, 0x210230, 0x220233, 0x220337,
    0x23033A, 0x24033E, 0x240341, 0x240344, 0x250347, 0x25044A, 0x26044D, 0x260450,
    0x270453, 0x270456, 0x280458, 0x28045B, 0x29045D, 0x2A0560, 0x2A0562, 0x2B0565,
    0x2C0567, 0x2C056A, 0x2C056D, 0x2C0571, 0x2B0676, 0x29067B, 0x250682, 0x1E0789,
    0x0F0793, 0x070E92, 0x07158D, 0x061A87, 0x061F82, 0x06227D, 0x062578, 0x052873,
    0x052B6E, 0x052D6A, 0x052F66, 0x053162, 0x04335E, 0x04355B, 0x043658, 0x043855,
    0x043952, 0x043B50, 0x043C4D, 0x043D4B, 0x043F49, 0x034047, 0x034146, 0x034244,
    0x034443, 0x034541, 0x034640, 0x03473E, 0x03483D, 0x04493B, 0x044A3A, 0x044C38,
    0x044D37, 0x044E35, 0x044F33, 0x045032, 0x045130, 0x04522E, 0x04542C, 0x04552A,
    0x045628, 0x045726, 0x045824, 0x045922, 0x045A20, 0x045B1E, 0x055D1C, 0x055E1A,
    0x055F18, 0x056016, 0x056114, 0x056212, 0x056310, 0x05640F, 0x05650E, 0x05660E,
    0x05680D, 0x05690D, 0x056A0C, 0x056B0B, 0x056C0A, 0x056D08, 0x056E06, 0x096F05,
    0x0F7005, 0x147105, 0x187205, 0x1D7306, 0x227306, 0x267406, 0x2B7506, 0x2F7606,
    0x337606, 0x387706, 0x3C7706, 0x407806, 0x457806, 0x497906, 0x4D7906, 0x527A06,
    0x567A06, 0x5A7A06, 0x5F7B06, 0x637B06, 0x677B06, 0x6B7B06, 0x6F7C06, 0x737C06,
    0x787C06, 0x7C7C06, 0x807C06, 0x857C06, 0x8A7B07, 0x8F7B07, 0x947A07, 0x997A07,
    0x9F7908, 0xA47808, 0xAA7608, 0xB07508, 0xB77309, 0xBD7109, 0xC46F09, 0xCA6C0A,
    0xD1690A, 0xD8660A, 0xDF620B, 0xE65E0B, 0xED590B, 0xF3540C, 0xF4561B, 0xF55824,
    0xF5592C, 0xF55B32, 0xF55D37, 0xF65F3C, 0xF66240, 0xF66443, 0xF66646, 0xF66849,
    0xF66A4C, 0xF76B51, 0xF76D55, 0xF76F5B, 0xF77060, 0xF87266, 0xF8736C, 0xF87573,
    0xF87679, 0xF9787F, 0xF97986, 0xF97B8C, 0xF97C92, 0xF97D98, 0xF97F9E, 0xF980A4,
    0xF982AA, 0xF983AF, 0xF984B5, 0xF985BA, 0xF987BF, 0xF988C4, 0xF989C9, 0xF98BCE,
    0xF98CD3, 0xFA8DD8, 0xFA8EDC, 0xFA90E0, 0xFA91E5, 0xFA92E9, 0xFA94EC, 0xFA95F0,
    0xFA96F4, 0xFA98F8, 0xF99AFA, 0xF69DFA, 0xF3A1FA, 0xF1A4FB, 0xEFA7FB, 0xEDAAFB,
    0xEBACFB, 0xE9AFFB, 0xE8B1FB, 0xE7B3FB, 0xE6B5FB, 0xE5B7FC, 0xE4B9FC, 0xE4BBFC,
    0xE3BDFC, 0xE3BFFC, 0xE3C0FC, 0xE3C2FC, 0xE3C4FC, 0xE3C5FC, 0xE3C7FC, 0xE4C8FC,
    0xE4CAFC, 0xE4CBFD, 0xE5CDFD, 0xE5CEFD, 0xE6D0FD, 0xE7D1FD, 0xE7D2FD, 0xE8D4FD,
    0xE9D5FD, 0xE9D6FD, 0xEAD8FD, 0xEAD9FD, 0xEADBFD, 0xEADCFD, 0xEBDEFD, 0xEBE0FD,
    0xEBE1FE, 0xEBE3FE, 0xEBE4FE, 0xEBE6FE, 0xEBE7FE, 0xEBE9FE, 0xEBEAFE, 0xECECFE,
    0xECEDFE, 0xECEFFE, 0xECF0FE, 0xEDF2FE, 0xEDF3FE, 0xEEF5FE, 0xEEF6FE, 0xEFF7FE,
    0xF0F9FE, 0xF1FAFE, 0xF2FBFE, 0xF4FCFE, 0xF6FDFF, 0xF8FEFF, 0xFBFFFF, 0xFFFFFF,
};
#endif

static char* create_image(const FastBoard::vertex_t* board, const float* data, int* len) {
#if 0
    float min = 0.0f;
    float max = 0.10f;
#else
    float min = 10e10;
    float max = -10e10;
    for (auto i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        auto v = data[i];
        if (v < min)
            min = v;
        if (v > max)
            max = v;
    }
#endif
    auto buf = std::unique_ptr<unsigned char[]>(new unsigned char[BOARD_SIZE * BOARD_SIZE * 3]);
    for (auto i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        auto v = (int)((data[i] - min) / (max - min) * 255.0f + 0.5f);
        auto n = std::min(std::max(v, 0), 255);
        buf[i * 3 + 0] = (color_map[n] >> 16) & 0xff;
        buf[i * 3 + 1] = (color_map[n] >> 8) & 0xff;
        buf[i * 3 + 2] = color_map[n] & 0xff;
    }
#if 0
    unsigned char *png = stbi_write_png_to_mem(buf.get(), 3 * BOARD_SIZE,
      BOARD_SIZE, BOARD_SIZE, 3, len);
#else
    const auto scale = 5;
    auto out = std::unique_ptr<unsigned char[]>(new unsigned char[BOARD_SIZE * BOARD_SIZE * 3 * scale * scale]);
    stbir_resize_uint8(buf.get(), BOARD_SIZE, BOARD_SIZE, 0,
      out.get(), BOARD_SIZE * scale, BOARD_SIZE * scale, 0, 3);
    const auto per_line = BOARD_SIZE * scale;
    for (auto y = 0; y < BOARD_SIZE; y++) {
        for (auto x = 0; x < BOARD_SIZE; x++) {
            auto idx = y * BOARD_SIZE + x;
            if (board[idx] == FastBoard::EMPTY) continue;
            unsigned char stone = board[idx] == FastBoard::BLACK ? 0x33 : 0xff;
            unsigned char stone2 = board[idx] == FastBoard::BLACK ? 0x0 : 0xcc;
            for (auto oy = 1; oy <= 3; oy++) {
                for (auto ox = 1; ox <= 3; ox++) {
                    out[((y * scale + oy) * per_line + x * scale + ox) * 3]     = stone;
                    out[((y * scale + oy) * per_line + x * scale + ox) * 3 + 1] = stone;
                    out[((y * scale + oy) * per_line + x * scale + ox) * 3 + 2] = stone;
                }
            }
            //out[((y * scale + 0) * per_line + x * scale + 2) * 3] = stone;
            //out[((y * scale + 0) * per_line + x * scale + 2) * 3 + 1] = stone;
            //out[((y * scale + 0) * per_line + x * scale + 2) * 3 + 2] = stone;
            //out[((y * scale + 2) * per_line + x * scale + 0) * 3] = stone;
            //out[((y * scale + 2) * per_line + x * scale + 0) * 3 + 1] = stone;
            //out[((y * scale + 2) * per_line + x * scale + 0) * 3 + 2] = stone;
            out[((y * scale + 2) * per_line + x * scale + 4) * 3] = stone2;
            out[((y * scale + 2) * per_line + x * scale + 4) * 3 + 1] = stone2;
            out[((y * scale + 2) * per_line + x * scale + 4) * 3 + 2] = stone2;
            out[((y * scale + 1) * per_line + x * scale + 4) * 3] = stone2;
            out[((y * scale + 1) * per_line + x * scale + 4) * 3 + 1] = stone2;
            out[((y * scale + 1) * per_line + x * scale + 4) * 3 + 2] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 2) * 3] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 2) * 3 + 1] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 2) * 3 + 2] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 3) * 3] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 3) * 3 + 1] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 3) * 3 + 2] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 4) * 3] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 4) * 3 + 1] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 4) * 3 + 2] = stone2;
        }
    }
    unsigned char *png = stbi_write_png_to_mem(out.get(), 3 * BOARD_SIZE * scale,
      BOARD_SIZE * scale, BOARD_SIZE * scale,
      3, len);
#endif
    return (char*) png;
}

static char* create_value_image(const float* data, const float* winrate, int* len) {
    assert(Network::VALUE_LAYER == 16 * 16);
    const auto padding = 4;
    const auto out_size = (BOARD_SIZE + padding) * 16;
    auto buf = std::unique_ptr<unsigned char[]>(new unsigned char[out_size * out_size * 3]);
    std::fill_n(buf.get(), out_size * out_size * 3, 255);
#if 0
    auto min = 10e10;
    auto max = -10e10;
    for (auto i = 0; i < BOARD_SIZE * BOARD_SIZE * 256; i++) {
      auto v = data[i];
      if (v < min)
        min = v;
      if (v > max)
        max = v;
    }
#elif 0
    const auto size = BOARD_SIZE * BOARD_SIZE * 256;
    auto sum = 0.0;
    auto sum2 = 0.0;
    for (auto i = 0; i < size; i++) {
      auto v = data[i];
      sum += v;
      sum2 += v * v;
    }
    auto ave = sum / size;
    auto variance = sum2 / size - ave * ave;
    auto sd = sqrt(variance);
    auto min = ave - sd * 2;
    auto max = ave + sd * 2;
#endif
    for (auto l = 0; l < 256; l++) {
        const auto d = data + l * BOARD_SIZE * BOARD_SIZE;
#if 1
        auto min = 10e10;
        auto max = -10e10;
        for (auto i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
            auto v = d[i];
            if (v < min)
                min = v;
            if (v > max)
                max = v;
        }
        if (min > -max)
          min = -max;
        if (max < -min)
          max = -min;
#endif
        const auto ox = l % 16 * (BOARD_SIZE + padding);
        const auto oy = l / 16 * (BOARD_SIZE + padding);
        for (auto i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
            auto ix = i % BOARD_SIZE;
            auto iy = i / BOARD_SIZE;
            auto v = (int)((d[i] - min) / (max - min) * 255.0f + 0.5f);
            auto n = std::min(std::max(v, 0), 255);
            buf[((oy + iy + 2) * out_size + ox + ix + 2) * 3 + 0] = (color_map[n] >> 16) & 0xff;
            buf[((oy + iy + 2) * out_size + ox + ix + 2) * 3 + 1] = (color_map[n] >> 8) & 0xff;
            buf[((oy + iy + 2) * out_size + ox + ix + 2) * 3 + 2] = color_map[n] & 0xff;
        }
        auto minw = -0.1f;
        auto maxw = 0.1f;
        for (auto i = 2; i < BOARD_SIZE + 2; i++) {
            auto v = (int)((winrate[l] - minw) / (maxw - minw) * 255.0f + 0.5f);
            auto n = std::min(std::max(v, 0), 255);
            buf[((oy)* out_size + ox + i) * 3 + 0] = (color_map[n] >> 16) & 0xff;
            buf[((oy)* out_size + ox + i) * 3 + 1] = (color_map[n] >> 8) & 0xff;
            buf[((oy)* out_size + ox + i) * 3 + 2] = color_map[n] & 0xff;
            buf[((oy + 1)* out_size + ox + i) * 3 + 0] = (color_map[n] >> 16) & 0xff;
            buf[((oy + 1)* out_size + ox + i) * 3 + 1] = (color_map[n] >> 8) & 0xff;
            buf[((oy + 1)* out_size + ox + i) * 3 + 2] = color_map[n] & 0xff;
            buf[((oy + 2 + BOARD_SIZE) * out_size + ox + i) * 3 + 0] = (color_map[n] >> 16) & 0xff;
            buf[((oy + 2 + BOARD_SIZE) * out_size + ox + i) * 3 + 1] = (color_map[n] >> 8) & 0xff;
            buf[((oy + 2 + BOARD_SIZE) * out_size + ox + i) * 3 + 2] = color_map[n] & 0xff;
            buf[((oy + 2 + BOARD_SIZE + 1) * out_size + ox + i) * 3 + 0] = (color_map[n] >> 16) & 0xff;
            buf[((oy + 2 + BOARD_SIZE + 1) * out_size + ox + i) * 3 + 1] = (color_map[n] >> 8) & 0xff;
            buf[((oy + 2 + BOARD_SIZE + 1) * out_size + ox + i) * 3 + 2] = color_map[n] & 0xff;

            buf[((oy + i) * out_size + ox) * 3 + 0] = (color_map[n] >> 16) & 0xff;
            buf[((oy + i) * out_size + ox) * 3 + 1] = (color_map[n] >> 8) & 0xff;
            buf[((oy + i) * out_size + ox) * 3 + 2] = color_map[n] & 0xff;
            buf[((oy + i) * out_size + ox + 1) * 3 + 0] = (color_map[n] >> 16) & 0xff;
            buf[((oy + i) * out_size + ox + 1) * 3 + 1] = (color_map[n] >> 8) & 0xff;
            buf[((oy + i) * out_size + ox + 1) * 3 + 2] = color_map[n] & 0xff;
            buf[((oy + i) * out_size + ox + 2 + BOARD_SIZE) * 3 + 0] = (color_map[n] >> 16) & 0xff;
            buf[((oy + i) * out_size + ox + 2 + BOARD_SIZE) * 3 + 1] = (color_map[n] >> 8) & 0xff;
            buf[((oy + i) * out_size + ox + 2 + BOARD_SIZE) * 3 + 2] = color_map[n] & 0xff;
            
        }
    }
#if 1
    unsigned char *png = stbi_write_png_to_mem(buf.get(), 3 * out_size,
     out_size, out_size, 3, len);
#else
    const auto scale = 2;
    auto out = std::unique_ptr<unsigned char[]>(new unsigned char[out_size * out_size * 3 * scale * scale]);
    stbir_resize_uint8(buf.get(), out_size, out_size, 0,
      out.get(), out_size * scale, out_size * scale, 0, 3);
    unsigned char *png = stbi_write_png_to_mem(out.get(), 3 * out_size * scale,
      out_size * scale, out_size * scale,
      3, len);
#endif
    return (char*) png;
}

void start_http_server() {
    extern int get_history_no(FastBoard::vertex_t color);
    extern const FastBoard::vertex_t* get_board_data_history(int history_pos);
    extern const float* get_policy_data_history(int history_pos);
    extern const float* get_value_data_history(int history_pos);
    extern const float* get_value2_data_history(int history_pos);
    extern const float* get_winrate_data_history(int history_pos);
    std::thread t([&]{
            using namespace httplib;
            Server svr;
            auto ret = svr.set_mount_point("/", cfg_server_dir.c_str());
            if (!ret) {
                printf("www directory not found\n");
                return;
            }
            svr.Get(R"(/policy/(b|w)/(\d+))", [](const Request& req, Response& res) {
                auto color = req.matches[1].str() == "b" ? FastBoard::BLACK : FastBoard::WHITE;
                auto history_no = get_history_no(color);
                auto num = std::atoi(req.matches[2].str().c_str());
                auto board = get_board_data_history(history_no);
                auto data = get_policy_data_history(history_no) + num * BOARD_SIZE * BOARD_SIZE;
                int len;
                auto png = create_image(board, data, &len);
                if (png == NULL) return;
                res.set_content((char*)png, len, "image/png");
                STBIW_FREE(png);
            });
            svr.Get(R"(/value/(b|w))", [](const Request& req, Response& res) {
                auto color = req.matches[1].str() == "b" ? FastBoard::BLACK : FastBoard::WHITE;
                auto history_no = get_history_no(color);
                auto num = std::atoi(req.matches[2].str().c_str());
                auto board = get_board_data_history(history_no);
                auto data = get_value_data_history(history_no);
                int len;
                auto png = create_image(board, data, &len);
                if (png == NULL) return;
                res.set_content((char*)png, len, "image/png");
                STBIW_FREE(png);
            });
            svr.Get(R"(/value2/(b|w))", [](const Request& req, Response& res) {
                auto color = req.matches[1].str() == "b" ? FastBoard::BLACK : FastBoard::WHITE;
                auto history_no = get_history_no(color);
                auto num = std::atoi(req.matches[2].str().c_str());
                //auto board = get_board_data_history(history_no);
                auto data = get_value2_data_history(history_no);
                auto data2 = get_winrate_data_history(history_no);
                int len;
                auto png = create_value_image(data, data2, &len);
                if (png == NULL) return;
                res.set_content((char*)png, len, "image/png");
                STBIW_FREE(png);
            });
            svr.listen("localhost", cfg_server_port);
    });
    t.detach();
}
