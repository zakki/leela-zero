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

#include <vector>
#include <algorithm>
#include <cassert>

#include "config.h"
#include "Ladder.h"

#include "Utils.h"

using namespace Utils;

const int LADDER_DEPTH_MAX = 100;
const int OK_DEPTH = 7;     // if ladder search needs deep depth, it may be ladder.

size_t s_root_movenum;
size_t s_clear_board_num;

Ladder::LadderStatus Ladder::ladder_status(const FastState & /*state*/) {

    Ladder::LadderStatus status;
/*
    const auto board = state.board;
    for (auto i = 0; i < BOARD_SIZE; i++) {
        for (auto j = 0; j < BOARD_SIZE; j++) {
            auto vertex = board.get_vertex(i, j);
            status[i][j] = NO_LADDER;
            if (ladder_capture(state, vertex)) {
                status[i][j] = CAPTURE;
            }
            if (ladder_escape(state, vertex)) {
                status[i][j] = ESCAPE;
            }
        }
    }
*/
    return status;
}

bool Ladder::ladder_capture(const FastState &state, int vertex, int *p_searched_depth, int group, int depth) {

    const auto &board = state.board;
    const auto capture_player = board.get_to_move();
    const auto escape_player = board.get_not_to_move();

    if (!state.is_move_legal(capture_player, vertex)) {
        if ( depth > *p_searched_depth ) *p_searched_depth = depth;
        return false;
    }

    // Assume that capture succeeds if it takes this long
    if (depth >= LADDER_DEPTH_MAX) {
        if ( depth > *p_searched_depth ) *p_searched_depth = depth;
        return true;
    }

    std::vector<int> groups_in_ladder;

    if (group == FastBoard::PASS) {
        // Check if there are nearby groups with 2 liberties
        for (int d = 0; d < 4; d++) {
            int n_vtx = board.get_state_neighbor(vertex, d);
            int n = board.get_state(n_vtx);
            if ((n == escape_player) && (board.get_liberties(n_vtx) == 2)) {
                auto parent = board.m_parent[n_vtx];
                if (std::find(groups_in_ladder.begin(), groups_in_ladder.end(), parent) == groups_in_ladder.end()) {
                    groups_in_ladder.emplace_back(parent);
                }
            }
        }
    } else {
        groups_in_ladder.emplace_back(group);
    }

    for (auto& group : groups_in_ladder) {
        auto state_copy = std::make_unique<FastState>(state);
        auto &board_copy = state_copy->board;

        state_copy->play_move(vertex);

        int escape = FastBoard::PASS;
        int newpos = group;
        do {
            for (int d = 0; d < 4; d++) {
                int stone = newpos + board_copy.m_dirs[d];
                // If the surrounding stones are in atari capture fails
                if (board_copy.m_state[stone] == capture_player) {
                    if (board_copy.get_liberties(stone) == 1) {
                        if ( depth > *p_searched_depth ) *p_searched_depth = depth;
                        return false;
                    }
                }
                // Possible move to escape
                if (board_copy.m_state[stone] == FastBoard::EMPTY) {
                    escape = stone;
                }
            }
            newpos = board_copy.m_next[newpos];
        } while (newpos != group);

        assert(escape != FastBoard::PASS);

        // If escaping fails the capture was successful
        if (!ladder_escape(*state_copy, escape, p_searched_depth, group, depth + 1)) {
            return true;
        }
    }

    if ( depth > *p_searched_depth ) *p_searched_depth = depth;
    return false;
}

bool Ladder::ladder_escape(const FastState &state, const int vertex, int *p_searched_depth, int group, int depth) {
    const auto &board = state.board;
    const auto escape_player = board.get_to_move();

    if (!state.is_move_legal(escape_player, vertex)) {
        if ( depth > *p_searched_depth ) *p_searched_depth = depth;
        return false;
    }

    // Assume that escaping failed if it takes this long
    if (depth >= LADDER_DEPTH_MAX) {
        if ( depth > *p_searched_depth ) *p_searched_depth = depth;
        return false;
    }

    std::vector<int> groups_in_ladder;

    if (group == FastBoard::PASS) {
        // Check if there are nearby groups with 1 liberties
        for (int d = 0; d < 4; d++) {
            int n_vtx = board.get_state_neighbor(vertex, d);
            int n = board.get_state(n_vtx);
            if ((n == escape_player) && (board.get_liberties(n_vtx) == 1)) {
                auto parent = board.m_parent[n_vtx];
                if (std::find(groups_in_ladder.begin(), groups_in_ladder.end(), parent) == groups_in_ladder.end()) {
                    groups_in_ladder.emplace_back(parent);
                }
            }
        }
    } else {
        groups_in_ladder.emplace_back(group);
    }

    for (auto& group : groups_in_ladder) {
        auto state_copy = std::make_unique<FastState>(state);
        auto &board_copy = state_copy->board;

        state_copy->play_move(vertex);

        if (board_copy.get_liberties(group) >= 3) {
            // Opponent can't atari on the next turn
            if ( depth > *p_searched_depth ) *p_searched_depth = depth;
            return true;
        }

        if (board_copy.get_liberties(group) == 1) {
            // Will get captured on the next turn
            if ( depth > *p_searched_depth ) *p_searched_depth = depth;
            return false;
        }

        // Still two liberties left, check for possible captures
        int newpos = group;
        do {
            for (int d = 0; d < 4; d++) {
                int empty = newpos + board_copy.m_dirs[d];
                if (board_copy.m_state[empty] == FastBoard::EMPTY) {
                    if (ladder_capture(*state_copy, empty, p_searched_depth, group, depth + 1)) {
                        // Got captured
                        if ( depth > *p_searched_depth ) *p_searched_depth = depth;
                        return false;
                    }
                }
            }
            newpos = board_copy.m_next[newpos];
        } while (newpos != group);

        // Ladder capture failed, escape succeeded
        if ( depth > *p_searched_depth ) *p_searched_depth = depth;
        return true;
    }

    if ( depth > *p_searched_depth ) *p_searched_depth = depth;
    return false;
}


int Ladder::ladder_maybe(const FastState &state, int vertex) {

    const auto &board = state.board;
    const auto to_move     = board.get_to_move();

    if (!state.is_move_legal(to_move, vertex)) {
        return NO_LADDER;
    }

    int group = FastBoard::PASS;

    // find ladder escape. libs = 1
    int emp4 = 0;
    int d;
    for (d = 0; d < 4; d++) {
        int n_vtx = board.get_state_neighbor(vertex, d);
        int n = board.get_state(n_vtx);
        if ( n == FastBoard::EMPTY ) emp4++;
//      if ( n == FastBoard::INVAL ) break;
        if ( n != to_move ) continue;
        int libs = board.get_liberties(n_vtx);
        if ( libs >= 4 ) break;
//      if ( libs >= 3 ) break;
        if ( libs != 1 ) continue;
        auto parent = board.m_parent[n_vtx];
        if ( group == FastBoard::PASS ) {
            group = parent;
        } else {
//          if ( group != parent ) break;
        }
    }
    if ( d == 4 && group != FastBoard::PASS && emp4 <= 2 ) {
//  if ( d == 4 && group != FastBoard::PASS && emp4 == 2 ) {
        int searched_depth = 0;
        bool ret = ladder_escape(state, vertex, &searched_depth);
//      myprintf("escape_search ret=%d,depth=%d,vtx=%s\n",ret,searched_depth,board.move_to_text(vertex).c_str());
        if ( ret == false && searched_depth >= OK_DEPTH ) {
            return CANNOT_ESCAPE;
        }
    }


    //
    group = FastBoard::PASS;
    emp4 = 0;
    for (d = 0; d < 4; d++) {
        int n_vtx = board.get_state_neighbor(vertex, d);
        int n = board.get_state(n_vtx);
        if ( n == FastBoard::EMPTY ) emp4++;
//      if ( n == FastBoard::INVAL ) break;
        if ( n != to_move ) continue;
        int libs = board.get_liberties(n_vtx);
        if ( libs >= 4 ) break;
        if ( libs == 1 ) continue;  // lib1 is already checked.
        auto parent = board.m_parent[n_vtx];
        if ( group == FastBoard::PASS ) {
            group = parent;
        } else {
//          if ( group != parent ) break;
        }
    }
    if ( d == 4 && group != FastBoard::PASS && emp4 <= 2 ) {
        int searched_depth = 0;
        bool ret = ladder_escape(state, vertex, &searched_depth, vertex);
//      myprintf("captured lib2 ret=%d,depth=%d,vtx=%s\n",ret,searched_depth,board.move_to_text(vertex).c_str());
        if ( ret == false && searched_depth >= OK_DEPTH ) {
            return CANNOT_ESCAPE;
        }
    }

#if 0
/*
OOOOO
OO..O ladder, but escape ok. killing eye.
OX..O
OOOOO
...........
...........
...........  "X" plays atari is bad move?
......OX...  "O" is not ladder.
..O..XOOXX.
......XXOO.
*/
    // find chase not ladder. libs = 2.
    const auto not_to_move = board.get_not_to_move();
    group = FastBoard::PASS;
    emp4 = 0;
    int same_col = 0;
    int stones = 0;
    const int LOOP = 4; // chase distanse
    for (d = 0; d < 4; d++) {
        int n_vtx = board.get_state_neighbor(vertex, d);
        int n = board.get_state(n_vtx);
        if ( n == FastBoard::EMPTY ) emp4++;
        if ( n == to_move     ) same_col++;
        if ( n != not_to_move ) continue;
        int libs = board.get_liberties(n_vtx);
        if ( libs != 2 ) break;
        auto parent = board.m_parent[n_vtx];
        if ( group != FastBoard::PASS ) {
            if ( group != parent ) break;
            continue;
        }
        int vtx2 = board.get_state_neighbor(n_vtx, d);
        if ( board.get_state(vtx2) != not_to_move ) continue;
        int vtx3 = board.get_state_neighbor(n_vtx, (d+1)&3);
        int vtx4 = board.get_state_neighbor(n_vtx, (d+3)&3);
        int n3 = board.get_state(vtx3);
        int n4 = board.get_state(vtx4);
        int dd = 0;
        if ( n3 == FastBoard::EMPTY && n4 == to_move && board.get_liberties(vtx4) == 2 ) dd = 3;
        if ( n4 == FastBoard::EMPTY && n3 == to_move && board.get_liberties(vtx3) == 2 ) dd = 1;
        if ( dd == 0 ) continue;
        int vtx_x = vtx2;
        int i,vtxr,vtxl,vtxd;
        for (i = 0; i < LOOP; i++) {
            bool flag = false;
            vtxr = board.get_state_neighbor(vtx_x, (d+dd  )&3);
            vtxl = board.get_state_neighbor(vtx_x, (d+dd+2)&3);
            if ( board.get_state(vtxl) == to_move     && board.get_liberties(vtxl) == 2 ) flag = true;
            if ( board.get_state(vtxr) == not_to_move                                   ) flag = true;
            if ( flag != true ) break;
            vtx_x = vtxr;
            vtxr = board.get_state_neighbor(vtx_x, (d+dd  )&3);
            vtxd = board.get_state_neighbor(vtx_x, (d     )&3);
            flag = false;
            if ( board.get_state(vtxr) == to_move     && board.get_liberties(vtxl) == 2 ) flag = true;
            if ( board.get_state(vtxd) == not_to_move                                   ) flag = true;
            if ( flag != true ) break;
            vtx_x = vtxd;
        }
        if ( i != LOOP ) continue;
        group = parent;
        stones = board.m_stones[group];
    }
    if ( d == 4 && group != FastBoard::PASS && same_col == 0 && emp4 == 3 ) {
        int searched_depth = 0;
        bool ret = ladder_capture(state, vertex, &searched_depth);
        myprintf("capture_search ret=%d,depth=%d,stones=%d,vtx=%s,LOOP=%d\n",ret,searched_depth,stones, board.move_to_text(vertex).c_str(),LOOP);
        if ( ret == false && searched_depth >= OK_DEPTH && stones >= 1 ) {
            return CANNOT_CAPTURE;
        }
        if ( ret == true  && searched_depth >= OK_DEPTH ) { // diffcult ladder. encourage a little
            return CAPTURE;
        }
    }
#endif

    return NO_LADDER;
}


static void print_columns() {
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (i < 25) {
            myprintf("%c ", (('a' + i < 'i') ? 'a' + i : 'a' + i + 1));
        } else {
            myprintf("%c ", (('A' + (i - 25) < 'I') ? 'A' + (i - 25) : 'A' + (i - 25) + 1));
        }
    }
    myprintf("\n");
}

void Ladder::display_ladders(const LadderStatus &status) {
    myprintf("\n   ");
    print_columns();
    for (int j = BOARD_SIZE-1; j >= 0; j--) {
        myprintf("%2d", j+1);
        myprintf(" ");
        for (int i = 0; i < BOARD_SIZE; i++) {
            if (status[i][j] == CAPTURE) {
                myprintf("C");
            } else if (status[i][j] == ESCAPE) {
                myprintf("E");
            } else if (FastBoard::starpoint(BOARD_SIZE, i, j)) {
                myprintf("+");
            } else {
                myprintf(".");
            }
            myprintf(" ");
        }
        myprintf("%2d\n", j+1);
    }
    myprintf("   ");
    print_columns();
    myprintf("\n");
}

void Ladder::display_ladders(const FastState &state) {
    display_ladders(ladder_status(state));
}

void Ladder::display_parent(const FastState &state) {
    myprintf("parent\n");
    for (auto i=0; i < NUM_INTERSECTIONS; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state.board.get_vertex(x, y);
        int k = state.board.m_parent[vertex];
        myprintf("%4d",k);
        if ( ((i+1) % BOARD_SIZE) == 0 ) myprintf("\n");
    }
    myprintf("next\n");
    for (auto i=0; i < NUM_INTERSECTIONS; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state.board.get_vertex(x, y);
        int k = state.board.m_next[vertex];
        myprintf("%4d",k);
        if ( ((i+1) % BOARD_SIZE) == 0 ) myprintf("\n");
    }
    myprintf("stones\n");
    for (auto i=0; i < NUM_INTERSECTIONS; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        auto vertex = state.board.get_vertex(x, y);
        if ( state.board.get_state(vertex) != FastBoard::EMPTY ) {
            vertex = state.board.m_parent[vertex];
        }
        int k = state.board.m_stones[vertex];
        myprintf("%4d",k);
        if ( ((i+1) % BOARD_SIZE) == 0 ) myprintf("\n");
    }
}
