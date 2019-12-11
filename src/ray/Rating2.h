#ifndef _RATING2_H_
#define _RATING2_H_

#include <string>
#include <random>

#include "GoBoard.h"
#include "SearchBoard.h"

namespace rating_v2 {
enum FEATURE1{
  F_CAPTURE1,
  F_CAPTURE2,
  F_CAPTURE3,
  F_CAPTURE4,
  F_SAVE,
  F_SAVE_CAPTURE,
  F_2POINT_CAPTURE,
  F_3POINT_CAPTURE,
  F_CAPTURE_AFTER_KO,
  F_SAVE_EXTENSION_SAFELY,
  F_SAVE_EXTENSION,
  F_CAPTURE_ESCAPABLE,
  F_MAX1,
};

enum FEATURE2 {
  F_SELF_ATARI_SMALL,
  F_SELF_ATARI_NAKADE,
  F_SELF_ATARI_LARGE,
  F_ATARI,
  F_2POINT_ATARI,
  F_2POINT_C_ATARI,
  F_3POINT_ATARI,
  F_3POINT_C_ATARI,
  F_2POINT_EXTENSION_SAFELY,
  F_2POINT_EXTENSION,
  F_3POINT_EXTENSION_SAFELY,
  F_3POINT_EXTENSION,
  F_MAX2,
};

const std::string po_features_name[F_MAX1 + F_MAX2] = {
  "CAPTURE1                ",
  "CAPTURE2                ",
  "CAPTURE3                ",
  "CAPTURE4                ",
  "SAVE                    ",
  "SAVE_CAPTURE            ",
  "2POINT_CAPTURE          ",
  "3POINT_CAPTURE          ",
  "CAPTURE_AFTER_KO        ",
  "SAVE_EXTENSION_SAFELY   ",
  "SAVE_EXTENSION          ",
  "CAPTURE_ESCAPABLE       ",
  "SELF_ATARI_SMALL        ",
  "SELF_ATARI_NAKADE       ",
  "SELF_ATARI_LARGE        ",
  "ATARI                   ",
  "2POINT_ATARI            ",
  "2POINT_C_ATARI          ",
  "3POINT_ATARI            ",
  "3POINT_C_ATARI          ",
  "2POINT_EXTENSION_SAFELY ",
  "2POINT_EXTENSION        ",
  "3POINT_EXTENSION_SAFELY ",
  "3POINT_EXTENSION        ",
};

const int TACTICAL_FEATURE_MAX = F_MAX1 + F_MAX2;
const int PREVIOUS_DISTANCE_MAX = 3;

const int PO_TACTICALS_MAX1 = (1 << F_MAX1);
const int PO_TACTICALS_MAX2 = (1 << F_MAX2);

// MD2パターンに入る手の数
const int UPDATE_NUM = 13;

const int F_MASK_MAX = 30;

// Simulation Parameter
const double NEIGHBOR_BIAS = 7.52598;
const double JUMP_BIAS = 4.63207;
const double PO_BIAS = 1.66542;

enum class rating_ladder_state_t {
  UNCHECKED,
  ILLEGAL,
  DEAD,
  ALIVE,
};

struct rating_context_t {
  game_info_t *game;
  search_game_info_t *search_game;
  position_t string_captured_pos[MAX_STRING * 2];
  rating_ladder_state_t string_captured[MAX_STRING * 2];

  explicit rating_context_t(game_info_t *src);
  ~rating_context_t();

  void clear();
};

////////////
//  変数  //
////////////

// ビットマスク
extern const unsigned int po_tactical_features_mask[F_MASK_MAX];

extern double po_temperature_set1;
extern double po_temperature_set2;
extern double po_temperature_pattern;
extern double po_offset;

////////////
//  関数  //
////////////

//  MD2に収まる座標の計算
void SetNeighbor( void );

//  初期化
void InitializeRating( void );

//  着手(Elo Rating)
int RatingMove( rating_context_t& ctx, int color, std::mt19937_64 *mt, long long *selected_rate );

//  レーティング
void Rating( rating_context_t& ctx, int color, long long *sum_rate, long long *sum_rate_row, long long *rate );

//  レーティング
void PartialRating( rating_context_t& ctx, int color, long long *sum_rate, long long *sum_rate_row, long long *rate );

//  現局面の評価値
void AnalyzePoRating( rating_context_t& ctx, int color, double rate[] );

rating_ladder_state_t GetLadderState( rating_context_t& ctx, int id, position_t lib, int color );
}

#endif
