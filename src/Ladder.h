/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Henrik Forsten

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
*/

#ifndef LADDER_H_INCLUDED
#define LADDER_H_INCLUDED

#include "FastState.h"

//void ladder_captures(const FastState &state);
//bool ladder_capture(const FastState &state, const int vertex);

class Ladder {

public:
    enum ladder_status_t : char {
        NO_LADDER = 0, CAPTURE = 1, ESCAPE = 2,
        CANNOT_ESCAPE = 3, CANNOT_CAPTURE = 4
    };

    using LadderStatus = std::array<std::array<ladder_status_t, BOARD_SIZE>, BOARD_SIZE>;

    static LadderStatus ladder_status(const FastState &state);

    static bool ladder_capture(const FastState &state, int vertex, int *searched_depth, int group = FastBoard::PASS, int depth = 0);
    static bool ladder_escape(const FastState &state, int vertex, int *searched_depth, int group = FastBoard::PASS, int depth = 0);
    static int ladder_maybe(const FastState &state, int vertex);

    static void display_ladders(const LadderStatus &status);
    static void display_ladders(const FastState &state);
    static void display_parent(const FastState &state);

};

#endif
