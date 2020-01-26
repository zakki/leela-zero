#include <algorithm>
#include <iostream>
#include <memory>

#include "Message.h"
#include "RayLadder.h"
#include "SearchBoard.h"
#include "Point.h"

using namespace std;

#define ALIVE true
#define DEAD  false


////////////////////////////////
//  現在の局面のシチョウ探索  //
////////////////////////////////
void
LadderExtension( const game_info_t *game, int color, uint8_t *ladder_pos )
{
  const string_t *string = game->string;
  std::unique_ptr<search_game_info_t> search_game;
  bool checked[BOARD_MAX] = { false };

  for (int i = 0; i < MAX_STRING; i++) {
    if (!string[i].flag ||
        string[i].size < 2 ||
	string[i].color != color) {
      continue;
    }
    // アタリから逃げる着手箇所
    int ladder = string[i].lib[0];

    bool flag = false;

    // アタリを逃げる手で未探索のものを確認
    if (!checked[ladder] && string[i].libs == 1) {
      if (!search_game)
        search_game.reset(new search_game_info_t(game));
      search_game_info_t *ladder_game = search_game.get();
      // 隣接する敵連を取って助かるかを確認
      int neighbor = string[i].neighbor[0];
      while (neighbor != NEIGHBOR_END && !flag) {
        if (string[neighbor].libs == 1) {
          if (IsLegal(game, string[neighbor].lib[0], color)) {
            PutStoneForSearch(ladder_game, string[neighbor].lib[0], color);
            int max_size = string[i].size;
            if (IsLadderCaptured(0, ladder_game, string[i].origin, FLIP_COLOR(color), max_size) == DEAD) {
              ladder_pos[string[neighbor].lib[0]] = min(max_size, 0xff);
            } else {
              flag = true;
            }
            Undo(ladder_game);
          }
        }
	neighbor = string[i].neighbor[neighbor];
      }

      // 取って助からない時は逃げてみる
      if (!flag) {
        if (IsLegal(game, ladder, color)) {
          PutStoneForSearch(ladder_game, ladder, color);
          int max_size = string[i].size;
          if (IsLadderCaptured(0, ladder_game, ladder, FLIP_COLOR(color), max_size) == DEAD) {
            ladder_pos[ladder] = min(max_size, 0xff);
          }
          Undo(ladder_game);
        }
      }
      checked[ladder] = true;
    }
  }
}


////////////////////
//  シチョウ探索  //
////////////////////
bool
IsLadderCaptured( const int depth, search_game_info_t *game, const int ren_xy, const int turn_color, int &max_size )
{
  const char *board = game->board;
  const string_t *string = game->string;
  const int str = game->string_id[ren_xy];
  int escape_color, capture_color;
  int escape_xy, capture_xy;
  int neighbor;
  bool result;

  if (depth >= 100 || game->moves >= MAX_RECORDS - 1) {
    return ALIVE;
  }

  if (board[ren_xy] == S_EMPTY) {
    return DEAD;
  } else if (string[str].libs >= 3) {
    return ALIVE;
  }

  escape_color = board[ren_xy];
  capture_color = FLIP_COLOR(escape_color);

  if (turn_color == escape_color) {
    // 周囲の敵連が取れるか確認し,
    // 取れるなら取って探索を続ける
    neighbor = string[str].neighbor[0];
    int local_max = max_size;
    while (neighbor != NEIGHBOR_END) {
      if (string[neighbor].libs == 1) {
	if (IsLegalForSearch(game, string[neighbor].lib[0], escape_color)) {
	  PutStoneForSearch(game, string[neighbor].lib[0], escape_color);
	  result = IsLadderCaptured(depth + 1, game, ren_xy, FLIP_COLOR(turn_color), local_max);
	  Undo(game);
	  if (result == ALIVE) {
	    return ALIVE;
	  }
	}
      }
      neighbor = string[str].neighbor[neighbor];
    }

    // 逃げる手を打ってみて探索を続ける
    escape_xy = string[str].lib[0];
    while (escape_xy != LIBERTY_END) {
      if (IsLegalForSearch(game, escape_xy, escape_color)) {
	PutStoneForSearch(game, escape_xy, escape_color);
	result = IsLadderCaptured(depth + 1, game, ren_xy, FLIP_COLOR(turn_color), local_max);
	Undo(game);
	if (result == ALIVE) {
	  return ALIVE;
	}
      }
      escape_xy = string[str].lib[escape_xy];
    }
    if (local_max > max_size)
      max_size = local_max;
    return DEAD;
  } else {
    if (string[str].libs == 1) {
      if (string[str].size > max_size)
        max_size = string[str].size;
      return DEAD;
    }
    // 追いかける側なのでアタリにする手を打ってみる
    capture_xy = string[str].lib[0];
    while (capture_xy != LIBERTY_END) {
      if (IsLegalForSearch(game, capture_xy, capture_color)) {
	PutStoneForSearch(game, capture_xy, capture_color);
	result = IsLadderCaptured(depth + 1, game, ren_xy, FLIP_COLOR(turn_color), max_size);
	Undo(game);
	if (result == DEAD) {
	  return DEAD;
	}
      }
      capture_xy = string[str].lib[capture_xy];
    }
    return ALIVE;
  }
}


//////////////////////////////////////////
//  助からないシチョウを逃げる手か判定  //
//////////////////////////////////////////
bool
CheckLadderExtension( const game_info_t *game, int color, int pos )
{
  const char *board = game->board;
  const string_t *string = game->string;
  const position_t *string_id = game->string_id;
  bool flag = false;

  if (board[pos] != color){
    return false;
  }

  int id = string_id[pos];

  int ladder = string[id].lib[0];

  if (string[id].libs == 1 &&
      IsLegal(game, ladder, color)) {
    std::unique_ptr<search_game_info_t> search_game(new search_game_info_t(game));
    search_game_info_t *ladder_game = search_game.get();
    PutStoneForSearch(ladder_game, ladder, color);
    int max_size = string[id].size;
    if (IsLadderCaptured(0, ladder_game, ladder, FLIP_COLOR(color), max_size) == DEAD) {
      flag = true;
    } else {
      flag = false;
    }
  }

  return flag;
}
