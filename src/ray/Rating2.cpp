#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <climits>
#include <algorithm>
#include <functional>
#include <string>
#include <numeric>
#include <set>

#include "RayLadder.h"
#include "Message.h"
#include "Nakade.h"
#include "Point.h"
#include "Rating2.h"
#include "Semeai.h"
#include "Utility.h"

using namespace std;

// パラメータのファイルを格納しているディレクトリのパス
extern char po_params_path[1024];

namespace rating_v2 {

rating_ladder_state_t
GetLadderState( rating_context_t& ctx, int id, position_t lib, int color )
{
  //static std::atomic<int> num_query;
  //static std::atomic<int> num_hit;
  //std::atomic_fetch_add(&num_query, 1);

  const game_info_t *game = ctx.game;
  const string_t *string = game->string;
  int ladder_no = -1;
  for (int i = 0; i < rating_context_t::num_move_cache; i++) {
    if (ctx.string_captured_pos[id * rating_context_t::num_move_cache + i] == lib) {
      ladder_no = i;
      break;
    }
  }
  if (ladder_no < 0) {
    for (int i = 0; i < rating_context_t::num_move_cache; i++) {
      if (ctx.string_captured_pos[id * rating_context_t::num_move_cache + i] == PASS) {
        ladder_no = i;
        break;
      }
    }
  }
  if (ladder_no < 0) {
    cerr << "BROKEN LADDER CACHE LIB:" << FormatMove(lib) << " STRING:" << FormatMove(string[id].origin);

    for (int i = 0; i < rating_context_t::num_move_cache; i++) {
      auto n = id * rating_context_t::num_move_cache + i;
      cerr << " LIB" << i << ":" << FormatMove(ctx.string_captured_pos[n]);
    }
    cerr << endl;
    PrintBoard(game);
    return rating_ladder_state_t::ILLEGAL;
  }

  const auto n = id * rating_context_t::num_move_cache + ladder_no;
  if (ctx.string_captured[n] != rating_ladder_state_t::UNCHECKED)
    return ctx.string_captured[n];

  /*
  std::atomic_fetch_add(&num_hit, 1);
  if (num_query % 10000 == 0)
    cerr << "LADDER CACHE " << (100.0 * num_hit / num_query) << endl;
  */

  if (IsLegalForSearch(ctx.search_game, lib, color)) {
    PutStoneForSearch(ctx.search_game, lib, color);
    int max_size = string[id].size;
    if (!IsLadderCaptured(0, ctx.search_game, string[id].origin, FLIP_COLOR(color), max_size)) {
      ctx.string_captured[n] = rating_ladder_state_t::DEAD;
      ctx.string_captured_pos[n] = lib;
    } else {
      ctx.string_captured[n] = rating_ladder_state_t::ALIVE;
      ctx.string_captured_pos[n] = lib;
    }
    Undo(ctx.search_game);
  } else {
    ctx.string_captured[n] = rating_ladder_state_t::ILLEGAL;
    ctx.string_captured_pos[n] = lib;
  }
  return ctx.string_captured[n];
}

rating_context_t::rating_context_t(game_info_t *src)
  : game(src), search_game(new search_game_info_t(src))
{
}

rating_context_t::~rating_context_t()
{
  delete search_game;
}

void
rating_context_t::clear()
{
  fill_n(string_captured, extent<decltype(string_captured)>::value, rating_ladder_state_t::UNCHECKED);
  fill_n(string_captured_pos, extent<decltype(string_captured_pos)>::value, PASS);
}

bool
rating_context_t::is_alive(int id) const
{
  for (auto i = 0; i < num_move_cache; i++) {
    if (string_captured[id * num_move_cache + i] == rating_ladder_state_t::ALIVE)
      return true;
  }
  return false;
}

bool
rating_context_t::is_dead(int id) const
{
  for (auto i = 0; i < num_move_cache; i++) {
    if (string_captured[id * num_move_cache + i] == rating_ladder_state_t::DEAD)
      return true;
  }
  return false;
}

}
