#ifndef _LADDER_H_
#define _LADDER_H_

#include "GoBoard.h"

struct search_game_info_t;

// 全ての連に対して逃げて助かるシチョウかどうか確認
void LadderExtension( const game_info_t *game, int color, uint8_t *ladder_pos );
// 戦術的特徴用の関数
bool CheckLadderExtension( const game_info_t *game, int color, int pos );
// シチョウ探索
bool IsLadderCaptured( const int depth, search_game_info_t *game, const int ren_xy, const int turn_color, int &max_size );
#endif
