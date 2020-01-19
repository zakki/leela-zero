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

////////////////
//    変数    //
////////////////

#if 0
// 3x3とMD2のパターンのγ値の積
static float po_pattern[MD2_MAX];
// 学習した着手距離の特徴
//float po_neighbor_orig[PREVIOUS_DISTANCE_MAX];
// 補正した着手距離の特徴
static float po_previous_distance[PREVIOUS_DISTANCE_MAX];
// 戦術的特徴1
static float po_tactical_set1[PO_TACTICALS_MAX1];
// 戦術的特徴2
static float po_tactical_set2[PO_TACTICALS_MAX2];

double po_temperature_set1 = 0.36;
double po_temperature_set2 = 0.68;
double po_temperature_pattern = 0.78;
double po_offset = -1000;
#endif

// ビットマスク
const unsigned int po_tactical_features_mask[F_MASK_MAX] = {
  0x00000001,  0x00000002,  0x00000004,  0x00000008,
  0x00000010,  0x00000020,  0x00000040,  0x00000080,
  0x00000100,  0x00000200,  0x00000400,  0x00000800,
  0x00001000,  0x00002000,  0x00004000,  0x00008000,
  0x00010000,  0x00020000,  0x00040000,  0x00080000,
  0x00100000,  0x00200000,  0x00400000,  0x00800000,
  0x01000000,  0x02000000,  0x04000000,  0x08000000,
  0x10000000,  0x20000000,
};


// MD2パターンが届く範囲
static int neighbor[UPDATE_NUM];

// コスミの位置
static int cross[4];

// 着手距離2, 3のγ値の補正
static double neighbor_bias = NEIGHBOR_BIAS;
// 着手距離4のγ値の補正
static double jump_bias = JUMP_BIAS;


// 隅のセキ
static set<unsigned int> seki_22_set[2];

//////////////////
//  関数の宣言  //
//////////////////

//  呼吸点が1つの連に対する特徴の判定
static void PoCheckFeaturesLib1( rating_context_t& ctx, const int color, const int id, position_t *update, int *update_num );

//  呼吸点が2つの連に対する特徴の判定
static void PoCheckFeaturesLib2( rating_context_t& ctx, const int color, const int id, position_t *update, int *update_num );

//  呼吸点が3つの連に対する特徴の判定
static void PoCheckFeaturesLib3( rating_context_t& ctx, const int color, const int id, position_t *update, int *update_num );

//  特徴の判定
static void PoCheckFeatures( rating_context_t& ctx, const int color, int previous_move, position_t *update, int *update_num );

//  劫を解消するトリの判定
static void PoCheckCaptureAfterKo( game_info_t *game, const int color, position_t *update, int *update_num );

//  自己アタリの判定
bool PoCheckSelfAtari( game_info_t *game, const int color, const position_t pos );

//  トリとアタリの判定
static void PoCheckCaptureAndAtari( rating_context_t& ctx, const int color, const position_t pos );

//  γ読み込み
//static void InputPOGamma( void );
//static void InputMD2( const char *filename, float *ap );

//  戦術的特徴の初期化
static void InitializePoTacticalFeaturesSet( void );


/////////////////
// 近傍の設定  //
/////////////////
void
SetNeighbor( void )
{
  neighbor[ 0] = -2 * board_size;
  neighbor[ 1] = - board_size - 1;
  neighbor[ 2] = - board_size;
  neighbor[ 3] = - board_size + 1;
  neighbor[ 4] = -2;
  neighbor[ 5] = -1;
  neighbor[ 6] = 0;
  neighbor[ 7] = 1;
  neighbor[ 8] = 2;
  neighbor[ 9] = board_size - 1;
  neighbor[10] = board_size;
  neighbor[11] = board_size + 1;
  neighbor[12] = 2 * board_size;

  cross[0] = -board_size - 1;
  cross[1] = -board_size + 1;
  cross[2] = board_size - 1;
  cross[3] = board_size + 1;
}

//////////////
//  初期化  //
//////////////
void
InitializeRating( void )
{
  // γ読み込み
  //InputPOGamma();
  // 戦術的特徴をまとめる
  //InitializePoTacticalFeaturesSet();
  // seki in corner
  {
    unsigned int seki_22_md2[] = { 3938730, 8133034, 12327338, 4004266, 8198570, 12392874, 4069802, 8264106, 12458410 };
    for (unsigned int md2 : seki_22_md2) {
      unsigned int transp[16];
      MD2Transpose16(md2, transp);
      for (int i = 0; i < 16; i++)
        seki_22_set[i / 8].insert(transp[i]);
    }
  }
}


////////////////////////////
//  戦術的特徴をまとめる  //
////////////////////////////
static void
InitializePoTacticalFeaturesSet( void )
{
#if 0
  int i;
  double rate;

  for (i = 0; i < PO_TACTICALS_MAX1; i++){
    rate = 1.0;

    if ((i & po_tactical_features_mask[F_SAVE_CAPTURE]) > 0) {
      rate *= po_tactical_features[F_SAVE_CAPTURE];
    } else if ((i & po_tactical_features_mask[F_SAVE_CAPTURE_SELF_ATARI]) > 0) {
      rate *= po_tactical_features[F_SAVE_CAPTURE_SELF_ATARI];
    } else if ((i & po_tactical_features_mask[F_CAPTURE_AFTER_KO]) > 0) {
      rate *= po_tactical_features[F_CAPTURE_AFTER_KO];
    } else if ((i & po_tactical_features_mask[F_2POINT_CAPTURE]) > 0) {
      rate *= po_tactical_features[F_2POINT_CAPTURE];
    } else if ((i & po_tactical_features_mask[F_3POINT_CAPTURE]) > 0) {
      rate *= po_tactical_features[F_3POINT_CAPTURE];
    }

    if ((i & po_tactical_features_mask[F_SAVE4]) > 0) {
      rate *= po_tactical_features[F_SAVE4];
    } else if ((i & po_tactical_features_mask[F_SAVE3]) > 0) {
      rate *= po_tactical_features[F_SAVE3];
    } else if ((i & po_tactical_features_mask[F_SAVE2]) > 0) {
      rate *= po_tactical_features[F_SAVE2];
    } else if ((i & po_tactical_features_mask[F_SAVE1]) > 0) {
      rate *= po_tactical_features[F_SAVE1];
    }

    if ((i & po_tactical_features_mask[F_CAPTURE4]) > 0) {
      rate *= po_tactical_features[F_CAPTURE4];
    } else if ((i & po_tactical_features_mask[F_CAPTURE3]) > 0) {
      rate *= po_tactical_features[F_CAPTURE3];
    } else if ((i & po_tactical_features_mask[F_CAPTURE2]) > 0) {
      rate *= po_tactical_features[F_CAPTURE2];
    } else if ((i & po_tactical_features_mask[F_CAPTURE1]) > 0) {
      rate *= po_tactical_features[F_CAPTURE1];
    }

    if ((i & po_tactical_features_mask[F_SAVE_EXTENSION_SAFELY]) > 0) {
      rate *= po_tactical_features[F_SAVE_EXTENSION_SAFELY];
    } else if ((i & po_tactical_features_mask[F_SAVE_EXTENSION]) > 0) {
      rate *= po_tactical_features[F_SAVE_EXTENSION];
    }

    po_tactical_set1[i] = (float)rate;
  }


  for (i = 0; i < PO_TACTICALS_MAX2; i++) {
    rate = 1.0;

    if ((i & po_tactical_features_mask[F_SELF_ATARI_SMALL]) > 0) {
      rate *= po_tactical_features[F_SELF_ATARI_SMALL + F_MAX1];
    } else if ((i & po_tactical_features_mask[F_SELF_ATARI_NAKADE]) > 0) {
      rate *= po_tactical_features[F_SELF_ATARI_NAKADE + F_MAX1];
    } else if ((i & po_tactical_features_mask[F_SELF_ATARI_LARGE]) > 0) {
      rate *= po_tactical_features[F_SELF_ATARI_LARGE + F_MAX1];
    }

    if ((i & po_tactical_features_mask[F_2POINT_ATARI]) > 0) {
      rate *= po_tactical_features[F_2POINT_ATARI + F_MAX1];
    } else if ((i & po_tactical_features_mask[F_3POINT_ATARI]) > 0) {
      rate *= po_tactical_features[F_3POINT_ATARI + F_MAX1];
    }
    if ((i & po_tactical_features_mask[F_ATARI1]) > 0) {
      rate *= po_tactical_features[F_ATARI1 + F_MAX1];
    }

    if ((i & po_tactical_features_mask[F_2POINT_EXTENSION_SAFELY]) > 0) {
      rate *= po_tactical_features[F_2POINT_EXTENSION_SAFELY + F_MAX1];
    } else if ((i & po_tactical_features_mask[F_3POINT_EXTENSION_SAFELY]) > 0) {
      rate *= po_tactical_features[F_3POINT_EXTENSION_SAFELY + F_MAX1];
    } else if ((i & po_tactical_features_mask[F_2POINT_EXTENSION]) > 0) {
      rate *= po_tactical_features[F_2POINT_EXTENSION + F_MAX1];
    } else if ((i & po_tactical_features_mask[F_3POINT_EXTENSION]) > 0) {
      rate *= po_tactical_features[F_3POINT_EXTENSION + F_MAX1];
    }

    if ((i & po_tactical_features_mask[F_THROW_IN_2]) > 0) {
      rate *= po_tactical_features[F_THROW_IN_2 + F_MAX1];
    }

    po_tactical_set2[i] = (float)rate;
  }
#endif
}

#if 0
//////////////////
//  MD2パターン  //
//////////////////
static long long
GetPoGamma(const game_info_t* game, int color, position_t pos, double scale)
{
  //cerr << "UPDATE " << FormatMove(pos) << endl;
  int md2 = MD2(game->pat, pos);
  if (color == S_WHITE)
    md2 = MD2Reverse(md2);
  int tf1 = game->tactical_features1[pos];
  int tf2 = game->tactical_features2[pos];

  if (po_offset <= -1000) {
    double pat = po_pattern[0];
    double tf1w = po_tactical_set1[0];
    double tf2w = po_tactical_set2[0];
    tf1w /= po_temperature_set1;
    tf2w /= po_temperature_set2;
    pat /= po_temperature_pattern;

    po_offset = 10 - (pat + tf1w + tf2w);
    cerr << "set po_offest = " << po_offset << endl;
  }

  double tf1w = po_tactical_set1[tf1];
  double tf2w = po_tactical_set2[tf2];
  double pat = po_pattern[md2];

  tf1w /= po_temperature_set1;
  tf2w /= po_temperature_set2;
  pat /= po_temperature_pattern;

  double gamma = exp(pat + po_offset) * scale + exp(tf1w + tf2w + po_offset);

  return max((long long)1, (long long) min(gamma, numeric_limits<long long>::max() / 100.0 / pure_board_max));
}



//////////////////////
//  着手( rating )  //
//////////////////////
int
RatingMove( rating_context_t& ctx, int color, std::mt19937_64 *mt, long long *selected_rate )
{
  game_info_t *game = ctx.game;
  long long *rate = game->rate[color - 1];
  long long *sum_rate_row = game->sum_rate_row[color - 1];
  long long *sum_rate = &game->sum_rate[color - 1];
  static char stone[] = { '+', 'B', 'W', '#' };

  // レートの部分更新
  PartialRating(ctx, color, sum_rate, sum_rate_row, rate);

  // 合法手を選択するまでループ
  while (true){
    if (*sum_rate == 0) return PASS;

    //rand_num = ((*mt)() % (*sum_rate)) + 1;
    uniform_int_distribution<long long> dist(1, *sum_rate);
    long long rand_num = dist(*mt);

    // 縦方向の位置を求める
    int y = board_start;
    while (rand_num > sum_rate_row[y]){
      rand_num -= sum_rate_row[y++];
    }

    // 横方向の位置を求める
    position_t pos = POS(board_start, y);
    do{
      long long r = rate[pos];
      rand_num -= r;
      if (rand_num <= 0) {
        if (selected_rate)
          (*selected_rate) = r;
        break;
      }
      pos++;
    } while (true);

    // 選ばれた手が合法手ならループを抜け出し
    // そうでなければその箇所のレートを0にし, 手を選びなおす
    if (IsLegalNotEye(game, pos, color)) {
#if 0
#if 1
      if (IsBadMove(game, pos, color) && rate[pos] > 1) {
        long long r = rate[pos];
        *sum_rate -= rate[pos];
        sum_rate_row[y] -= rate[pos];
        rate[pos] = 1;
        *sum_rate += rate[pos];
        sum_rate_row[board_y[pos]] += rate[pos];
        //if (game->moves < 250) { PrintBoard(game); cerr << "RETRY " << stone[color] << " " << r << " " << FormatMove(pos) << endl; }
        continue;
      }
#endif
      int replace_num = 0;
      int replace[8];
      if (ReplaceMove(game, pos, color, replace, &replace_num)) {
        if (replace_num > 0) {
          int rep = replace[(*mt)() % replace_num];
          if (IsLegalNotEye(game, rep, color)) {
            //if (game->moves < 300) { PrintBoard(game); cerr << "REPLACE " << stone[color] << " " << FormatMove(pos) << " -> " << FormatMove(rep) << endl; }
            return rep;
          }
        }
      }
#endif
      return pos;
    } else {
      *sum_rate -= rate[pos];
      sum_rate_row[y] -= rate[pos];
      rate[pos] = 0;
    }
  }
}
#endif


////////////////////////////
//  12近傍の座標を求める  //
////////////////////////////
static void
Neighbor12( int previous_move, position_t distance_2[], position_t distance_3[], position_t distance_4[] )
{
  // 着手距離2の座標
  distance_2[0] = previous_move + neighbor[ 2];
  distance_2[1] = previous_move + neighbor[ 5];
  distance_2[2] = previous_move + neighbor[ 7];
  distance_2[3] = previous_move + neighbor[10];

  // 着手距離3の座標
  distance_3[0] = previous_move + neighbor[ 1];
  distance_3[1] = previous_move + neighbor[ 3];
  distance_3[2] = previous_move + neighbor[ 9];
  distance_3[3] = previous_move + neighbor[11];

  // 着手距離4の座標
  distance_4[0] = previous_move + neighbor[ 0];
  distance_4[1] = previous_move + neighbor[ 4];
  distance_4[2] = previous_move + neighbor[ 8];
  distance_4[3] = previous_move + neighbor[12];
}


#if 0
//////////////////////////////
//  直前の着手の周辺の更新  //
//////////////////////////////
static void
NeighborUpdate( rating_context_t& ctx, int color, long long *sum_rate, long long *sum_rate_row, long long *rate, position_t *update, bool *flag, int index )
{
  game_info_t *game = ctx.game;
  double bias[4];
  position_t pm1 = game->record[game->moves - 1].pos;

  bias[0] = bias[1] = bias[2] = bias[3] = 1.0;

  // 盤端での特殊処理
  if (index == 1) {
    int pos = game->record[game->moves - 1].pos;
    if ((border_dis_x[pos] == 1 && border_dis_y[pos] == 2) ||
        (border_dis_x[pos] == 2 && border_dis_y[pos] == 1)) {
      for (int i = 0; i < 4; i++) {
        if ((border_dis_x[update[i]] == 1 && border_dis_y[update[i]] == 2) ||
            (border_dis_x[update[i]] == 2 && border_dis_y[update[i]] == 1)) {
          bias[i] = 10.0;
          break;
        }
      }
    }
  }

  for (int i = 0; i < 4; i++) {
    position_t pos = update[i];
    if (game->candidates[pos]){
      if (flag[pos] && bias[i] == 1.0) continue;
      bool self_atari_flag = PoCheckSelfAtari(game, color, pos);

      // 元あったレートを消去
      *sum_rate -= rate[pos];
      sum_rate_row[board_y[pos]] -= rate[pos];

      if (!self_atari_flag){
        rate[pos] = 0;
      } else {
        PoCheckCaptureAndAtari(ctx, color, pos);

        double gamma = 1;
        if (pm1 != PASS) {
          int dis = DIS(pos, pm1);
          if (dis < 5) {
            gamma *= po_previous_distance[dis - 2];
          }
        }
        //gamma *= po_previous_distance[index];
        gamma *= bias[i];
        rate[pos] = GetPoGamma(game, color, pos, gamma);

        // 新たに計算したレートを代入
        *sum_rate += rate[pos];
        sum_rate_row[board_y[pos]] += rate[pos];
      }

      game->tactical_features1[pos] = 0;
      game->tactical_features2[pos] = 0;
    }
    flag[pos] = true;
  }
}


//////////////////////////
//  ナカデの急所の更新  //
//////////////////////////
static void
NakadeUpdate( rating_context_t& ctx, int color, long long *sum_rate, long long *sum_rate_row, long long *rate, position_t *nakade_pos, int nakade_num, bool *flag, position_t pm1 )
{
  game_info_t *game = ctx.game;
  for (int i = 0; i < nakade_num; i++) {
    position_t pos = nakade_pos[i];
    if (pos != NOT_NAKADE && game->candidates[pos]){
      bool self_atari_flag = PoCheckSelfAtari(game, color, pos);

      // 元あったレートを消去
      *sum_rate -= rate[pos];
      sum_rate_row[board_y[pos]] -= rate[pos];

      if (!self_atari_flag) {
        rate[pos] = 0;
      } else {
        PoCheckCaptureAndAtari(ctx, color, pos);
        int dis = DIS(pm1, pos);
        double gamma;
        if (dis < 5) {
          gamma = 10000.0 * po_previous_distance[dis - 2];
        } else {
          gamma = 10000.0;
        }
        rate[pos] = GetPoGamma(game, color, pos, gamma);

        // 新たに計算したレートを代入
        *sum_rate += rate[pos];
        sum_rate_row[board_y[pos]] += rate[pos];
      }

      game->tactical_features1[pos] = 0;
      game->tactical_features2[pos] = 0;
      flag[pos] = true;
    }
  }
}


////////////////////
//  レートの更新  //
////////////////////
void
OtherUpdate( rating_context_t& ctx, int color, long long *sum_rate, long long *sum_rate_row, long long *rate, int update_num, position_t *update, bool *flag )
{
  game_info_t *game = ctx.game;
  for (int i = 0; i < update_num; i++) {
    position_t pos = update[i];
    if (flag[pos]) continue;

    if (game->candidates[pos]) {
      bool self_atari_flag = PoCheckSelfAtari(game, color, pos);

      // 元あったレートを消去
      *sum_rate -= rate[pos];
      sum_rate_row[board_y[pos]] -= rate[pos];

      // パターン、戦術的特徴、距離のγ値
      if (!self_atari_flag) {
        rate[pos] = 0;
      } else {
        PoCheckCaptureAndAtari(ctx, color, pos);
        rate[pos] = GetPoGamma(game, color, pos, 1);

        // 新たに計算したレートを代入
        *sum_rate += rate[pos];
        sum_rate_row[board_y[pos]] += rate[pos];
      }

      game->tactical_features1[pos] = 0;
      game->tactical_features2[pos] = 0;
    }
    // 更新済みフラグを立てる
    flag[pos] = true;
  }
}


/////////////////////////////////
//  MD2パターンの範囲内の更新  //
/////////////////////////////////
void
Neighbor12Update( rating_context_t& ctx, int color, long long *sum_rate, long long *sum_rate_row, long long *rate, int update_num, position_t *update, bool *flag )
{
  game_info_t *game = ctx.game;
  for (int i = 0; i < update_num; i++) {
    for (int j = 0; j < UPDATE_NUM; j++) {
      position_t pos = update[i] + neighbor[j];
      if (flag[pos]) continue;

      if (game->candidates[pos]) {
        bool self_atari_flag = PoCheckSelfAtari(game, color, pos);

        // 元あったレートを消去
        *sum_rate -= rate[pos];
        sum_rate_row[board_y[pos]] -= rate[pos];

        // パターン、戦術的特徴、距離のγ値
        if (!self_atari_flag){
          rate[pos] = 0;
        } else {
          PoCheckCaptureAndAtari(ctx, color, pos);
          rate[pos] = GetPoGamma(game, color, pos, 1);

          // 新たに計算したレートを代入
          *sum_rate += rate[pos];
          sum_rate_row[board_y[pos]] += rate[pos];
        }

        game->tactical_features1[pos] = 0;
        game->tactical_features2[pos] = 0;
      }
      // 更新済みフラグを立てる
      flag[pos] = true;
    }
  }
}
#endif


#if 0
////////////////
//  部分更新  //
////////////////
void
PartialRating( rating_context_t& ctx, int color, long long *sum_rate, long long *sum_rate_row, long long *rate )
{
  game_info_t *game = ctx.game;
  position_t pm1 = PASS, pm2 = PASS, pm3 = PASS;
  position_t distance_2[4], distance_3[4], distance_4[4];
  bool flag[BOARD_MAX] = { false };
  position_t *update_pos = game->update_pos[color];
  int *update_num = &game->update_num[color];
  int other = FLIP_COLOR(color);
  position_t nakade_pos[4] = { 0 };
  int nakade_num = 0;
  int prev_feature = game->update_num[color];
  position_t prev_feature_pos[PURE_BOARD_MAX];

  for (int i = 0; i < prev_feature; i++){
    prev_feature_pos[i] = update_pos[i];
  }

  fill_n(ctx.string_captured, extent<decltype(ctx.string_captured)>::value, rating_ladder_state_t::UNCHECKED);
  fill_n(ctx.string_captured_pos, extent<decltype(ctx.string_captured_pos)>::value, PASS);

  *update_num = 0;

  pm1 = game->record[game->moves - 1].pos;
  if (game->moves > 2) pm2 = game->record[game->moves - 2].pos;
  if (game->moves > 3) pm3 = game->record[game->moves - 3].pos;

  if (game->ko_move == game->moves - 2){
    PoCheckCaptureAfterKo(game, color, update_pos, update_num);
  }

  if (pm1 != PASS) {
    Neighbor12(pm1, distance_2, distance_3, distance_4);
    PoCheckFeatures(ctx, color, pm1, update_pos, update_num);

    SearchNakade(game, &nakade_num, nakade_pos);
    NakadeUpdate(ctx, color, sum_rate, sum_rate_row, rate, nakade_pos, nakade_num, flag, pm1);
    // 着手距離2の更新
    NeighborUpdate(ctx, color, sum_rate, sum_rate_row, rate, distance_2, flag, 0);
    // 着手距離3の更新
    NeighborUpdate(ctx, color, sum_rate, sum_rate_row, rate, distance_3, flag, 1);
    // 着手距離4の更新
    NeighborUpdate(ctx, color, sum_rate, sum_rate_row, rate, distance_4, flag, 2);

  }

  // 2手前の着手の12近傍の更新
  if (pm2 != PASS) {
    PoCheckFeatures(ctx, color, pm2, update_pos, update_num);
    Neighbor12Update(ctx, color, sum_rate, sum_rate_row, rate, 1, &pm2, flag);
  }
  // 3手前の着手の12近傍の更新
  if (pm3 != PASS) {
    PoCheckFeatures(ctx, color, pm3, update_pos, update_num);
    Neighbor12Update(ctx, color, sum_rate, sum_rate_row, rate, 1, &pm3, flag);
  }

#if 0
  // 3, 5, 7
  for (int i = 0; i < 3; i++) {
    int n = (i + 1) * 2 + 1;
    if (game->moves > n)
      PoCheckFeatures(ctx, color, game->record[game->moves - n].pos, update_pos, update_num);
  }
#endif

  // 以前の着手で戦術的特徴が現れた箇所の更新
  OtherUpdate(ctx, color, sum_rate, sum_rate_row, rate, prev_feature, prev_feature_pos, flag);
  // 最近の自分の着手の時に戦術的特徴が現れた箇所の更新
  OtherUpdate(ctx, color, sum_rate, sum_rate_row, rate, game->update_num[color], game->update_pos[color], flag);
  // 最近の相手の着手の時に戦術的特徴が現れた箇所の更新
  OtherUpdate(ctx, color, sum_rate, sum_rate_row, rate, game->update_num[other], game->update_pos[other], flag);
  // 自分の着手で石を打ち上げた箇所のとその周囲の更新
  Neighbor12Update(ctx, color, sum_rate, sum_rate_row, rate, game->capture_num[color], game->capture_pos[color], flag);
  // 相手の着手で石を打ち上げられた箇所とその周囲の更新
  Neighbor12Update(ctx, color, sum_rate, sum_rate_row, rate, game->capture_num[other], game->capture_pos[other], flag);

}


////////////////////
//  レーティング  //
////////////////////
void
Rating( rating_context_t& ctx, int color, long long *sum_rate, long long *sum_rate_row, long long *rate )
{
  game_info_t *game = ctx.game;
  int pm1 = PASS;
  int update_num = 0;
  position_t update_pos[PURE_BOARD_MAX];
  const string_t *string = game->string;

  pm1 = game->record[game->moves - 1].pos;

  fill_n(ctx.string_captured, extent<decltype(ctx.string_captured)>::value, rating_ladder_state_t::UNCHECKED);
  fill_n(ctx.string_captured_pos, extent<decltype(ctx.string_captured_pos)>::value, PASS);

  PoCheckFeatures(ctx, color, pm1, update_pos, &update_num);
  if (game->ko_move == game->moves - 2) {
    PoCheckCaptureAfterKo(game, color, update_pos, &update_num);
  }

  // Update semeai features
  for (int id = 0; id < MAX_STRING; id++) {
    if (!string[id].flag || string[id].color != color)
      continue;
    update_num = 0;
    if (string[id].libs == 1) {
      PoCheckFeaturesLib1(ctx, color, id, update_pos, &update_num);
    } else if (string[id].libs == 2) {
      PoCheckFeaturesLib2(ctx, color, id, update_pos, &update_num);
    } else if (string[id].libs == 3) {
      PoCheckFeaturesLib3(ctx, color, id, update_pos, &update_num);
    }
  }

  for (int i = 0; i < pure_board_max; i++) {
    int pos = onboard_pos[i];
    if (game->candidates[pos] && IsLegalNotEye(game, pos, color)) {
      bool self_atari_flag = PoCheckSelfAtari(game, color, pos);
      PoCheckCaptureAndAtari(ctx, color, pos);

      if (!self_atari_flag) {
        rate[pos] = 0;
      } else {
        double gamma = 1;
        if (pm1 != PASS) {
          int dis = DIS(pos, pm1);
          if (dis < 5) {
            gamma *= po_previous_distance[dis - 2];
          }
        }
        rate[pos] = GetPoGamma(game, color, pos, gamma);
      }

      *sum_rate += rate[pos];
      sum_rate_row[board_y[pos]] += rate[pos];

      game->tactical_features1[pos] = 0;
      game->tactical_features2[pos] = 0;
    }
  }
}
#endif


rating_ladder_state_t
GetLadderState( rating_context_t& ctx, int id, position_t lib, int color )
{
  //static std::atomic<int> num_query;
  //static std::atomic<int> num_hit;
  //std::atomic_fetch_add(&num_query, 1);

  const game_info_t *game = ctx.game;
  const string_t *string = game->string;
  int ladder_no = 0;
  if (ctx.string_captured_pos[id * 2 + 0] == lib) {
    ladder_no = 0;
  } else if (ctx.string_captured_pos[id * 2 + 1] == lib) {
    ladder_no = 1;
  } else if (ctx.string_captured_pos[id * 2 + 0] == PASS) {
    ladder_no = 0;
  } else if (ctx.string_captured_pos[id * 2 + 1] == PASS) {
    ladder_no = 1;
  } else {
    cerr << "BROKEN LADDER CACHE LIB:" << FormatMove(lib) << " STRING:" << FormatMove(string[id].origin)
       << " LIB1:" << FormatMove(ctx.string_captured_pos[id * 2 + 0])
       << " LIB2:" << FormatMove(ctx.string_captured_pos[id * 2 + 1])
         << endl;
    PrintBoard(game);
    return rating_ladder_state_t::ILLEGAL;
  }

  if (ctx.string_captured[id * 2 + ladder_no] != rating_ladder_state_t::UNCHECKED)
    return ctx.string_captured[id * 2 + ladder_no];

  /*
  std::atomic_fetch_add(&num_hit, 1);
  if (num_query % 10000 == 0)
    cerr << "LADDER CACHE " << (100.0 * num_hit / num_query) << endl;
  */

  if (IsLegalForSearch(ctx.search_game, lib, color)) {
    PutStoneForSearch(ctx.search_game, lib, color);
    int max_size = string[id].size;
    if (!IsLadderCaptured(0, ctx.search_game, string[id].origin, FLIP_COLOR(color), max_size)) {
      ctx.string_captured[id * 2 + ladder_no] = rating_ladder_state_t::DEAD;
      ctx.string_captured_pos[id * 2 + ladder_no] = lib;
    } else {
      ctx.string_captured[id * 2 + ladder_no] = rating_ladder_state_t::ALIVE;
      ctx.string_captured_pos[id * 2 + ladder_no] = lib;
    }
    Undo(ctx.search_game);
  } else {
    ctx.string_captured[id * 2 + ladder_no] = rating_ladder_state_t::ILLEGAL;
    ctx.string_captured_pos[id * 2 + ladder_no] = lib;
  }
  return ctx.string_captured[id * 2 + ladder_no];
}

/////////////////////////////////////////
//  呼吸点が1つの連に対する特徴の判定  //
/////////////////////////////////////////
static void
PoCheckFeaturesLib1( rating_context_t& ctx, const int color, const int id, position_t *update, int *update_num )
{
  game_info_t *game = ctx.game;
  const char *board = game->board;
  const position_t *string_id = game->string_id;
  const string_t *string = game->string;
  int lib, liberty;
  const int other = FLIP_COLOR(color);

  // 呼吸点が1つになった連の呼吸点を取り出す
  lib = string[id].lib[0];
  liberty = lib;

#if 1
  // 呼吸点の上下左右が敵石に接触しているか確認
  /*
  bool contact = false;
  if (board[NORTH(lib)] == other) contact = true;
  if (board[ EAST(lib)] == other) contact = true;
  if (board[ WEST(lib)] == other) contact = true;
  if (board[SOUTH(lib)] == other) contact = true;
  */

  rating_ladder_state_t state = GetLadderState(ctx, id, lib, color);
  if (state == rating_ladder_state_t::DEAD) {
    game->tactical_features1[lib] |= po_tactical_features_mask[F_SAVE_EXTENSION];
#if 0
    cerr << "EXTENSION " << FormatMove(lib) << " " << FormatMove(string[id].origin) << endl;
    PrintBoard(game);
#endif
  } else if (state == rating_ladder_state_t::ALIVE) {
    game->tactical_features1[lib] |= po_tactical_features_mask[F_SAVE_EXTENSION_SAFELY];
#if 0
    cerr << "EXTENSION SAFELY " << FormatMove(lib) << " " << FormatMove(string[id].origin) << endl;
    PrintBoard(game);
#endif
  }
#endif

  game->tactical_features1[lib] |= po_tactical_features_mask[F_SAVE];

  // レートの更新対象に入れる
  update[(*update_num)++] = lib;

  // 敵連を取ることによって連を助ける手の特徴の判定
  // 自分の連の大きさと敵の連の大きさで特徴を判定
  int neighbor = string[id].neighbor[0];
  while (neighbor != NEIGHBOR_END) {
    if (string[neighbor].libs == 1) {
      lib = string[neighbor].lib[0];
      game->tactical_features1[lib] |= po_tactical_features_mask[F_SAVE_CAPTURE];
      update[(*update_num)++] = lib;
    }
    neighbor = string[id].neighbor[neighbor];
  }
}


/////////////////////////////////////////
//  呼吸点が2つの連に対する特徴の判定  //
/////////////////////////////////////////
static void
PoCheckFeaturesLib2( rating_context_t& ctx, const int color, const int id, position_t *update, int *update_num )
{
  game_info_t *game = ctx.game;
  const position_t *string_id = game->string_id;
  const string_t *string = game->string;
  const char *board = game->board;
  int lib1, lib2;

  // 呼吸点が2つになった連の呼吸点を取り出す
  lib1 = string[id].lib[0];
  lib2 = string[id].lib[lib1];

  rating_ladder_state_t state1 = GetLadderState(ctx, id, lib1, color);
  if (state1 == rating_ladder_state_t::DEAD) {
    game->tactical_features2[lib1] |= po_tactical_features_mask[F_2POINT_EXTENSION];
  } else if (state1 == rating_ladder_state_t::ALIVE) {
    game->tactical_features2[lib1] |= po_tactical_features_mask[F_2POINT_EXTENSION_SAFELY];
  }

  rating_ladder_state_t state2 = GetLadderState(ctx, id, lib2, color);
  if (state2 == rating_ladder_state_t::DEAD) {
    game->tactical_features2[lib2] |= po_tactical_features_mask[F_2POINT_EXTENSION];
  } else if (state2 == rating_ladder_state_t::ALIVE) {
    game->tactical_features2[lib2] |= po_tactical_features_mask[F_2POINT_EXTENSION_SAFELY];
  }

  /*
  // 呼吸点の周囲が空点3つ, または呼吸点が3つ以上の自分の連に接続できるかで特徴を判定
  if (nb4_empty[Pat3(game->pat, lib1)] == 3 ||
      (board[NORTH(lib1)] == color && string_id[NORTH(lib1)] != id &&
       string[string_id[NORTH(lib1)]].libs >= 3) ||
      (board[ WEST(lib1)] == color && string_id[ WEST(lib1)] != id &&
       string[string_id[WEST(lib1)]].libs >= 3) ||
      (board[ EAST(lib1)] == color && string_id[ EAST(lib1)] != id &&
       string[string_id[EAST(lib1)]].libs >= 3) ||
      (board[SOUTH(lib1)] == color && string_id[SOUTH(lib1)] != id &&
       string[string_id[SOUTH(lib1)]].libs >= 3)) {
    game->tactical_features2[lib1] |= po_tactical_features_mask[F_2POINT_EXTENSION_SAFELY];
  } else {
    game->tactical_features2[lib1] |= po_tactical_features_mask[F_2POINT_EXTENSION];
  }

  // 呼吸点の周囲が空点3つ, または呼吸点が3つ以上の自分の連に接続できるかで特徴を判定
  if (nb4_empty[Pat3(game->pat, lib2)] == 3 ||
      (board[NORTH(lib2)] == color && string_id[NORTH(lib2)] != id &&
       string[string_id[NORTH(lib2)]].libs >= 3) ||
      (board[ WEST(lib2)] == color && string_id[ WEST(lib2)] != id &&
       string[string_id[WEST(lib2)]].libs >= 3) ||
      (board[ EAST(lib2)] == color && string_id[ EAST(lib2)] != id &&
       string[string_id[EAST(lib2)]].libs >= 3) ||
      (board[SOUTH(lib2)] == color && string_id[SOUTH(lib2)] != id &&
       string[string_id[SOUTH(lib2)]].libs >= 3)) {
    game->tactical_features2[lib2] |= po_tactical_features_mask[F_2POINT_EXTENSION_SAFELY];
  } else {
    game->tactical_features2[lib2] |= po_tactical_features_mask[F_2POINT_EXTENSION];
  }
  */

  PoCheckSelfAtari(game, color, lib2);

  // レートの更新対象に入れる
  update[(*update_num)++] = lib1;
  update[(*update_num)++] = lib2;

  // 呼吸点が2つになった連の周囲の敵連を調べる
  // 1. 呼吸点が1つの敵連
  // 2. 呼吸点が2つの敵連
  // それぞれに対して, 特徴を判定する
  int neighbor = string[id].neighbor[0];
  while (neighbor != NEIGHBOR_END) {
    if (string[neighbor].libs == 1) {
      lib1 = string[neighbor].lib[0];
      update[(*update_num)++] = lib1;
      game->tactical_features1[lib1] |= po_tactical_features_mask[F_2POINT_CAPTURE];
    } else if (string[neighbor].libs == 2) {
      lib1 = string[neighbor].lib[0];
      lib2 = string[neighbor].lib[lib1];
      update[(*update_num)++] = lib1;
      update[(*update_num)++] = lib2;

      rating_ladder_state_t c_state1 = GetLadderState(ctx, neighbor, lib1, color);
      if (c_state1 == rating_ladder_state_t::DEAD
        && state1 == rating_ladder_state_t::DEAD
        && state2 == rating_ladder_state_t::DEAD) {
        game->tactical_features2[lib1] |= po_tactical_features_mask[F_2POINT_C_ATARI];
#if 0
        cerr << "2POINT_C_ATARI " << FormatMove(lib1)
             << " CAPTURE:" << FormatMove(string[neighbor].origin)
             << " HELP:" << FormatMove(string[id].origin)
             << endl;
        PrintBoard(game);
#endif
      } else if (c_state1 == rating_ladder_state_t::DEAD || c_state1 == rating_ladder_state_t::ALIVE) {
        game->tactical_features2[lib1] |= po_tactical_features_mask[F_2POINT_ATARI];
#if 0
        cerr << "2POINT_ATARI " << FormatMove(lib1)
             << " CAPTURE:" << FormatMove(string[neighbor].origin)
             << " HELP:" << FormatMove(string[id].origin)
             << endl;
        PrintBoard(game);
#endif
      }

      rating_ladder_state_t c_state2 = GetLadderState(ctx, neighbor, lib2, color);
      if (c_state2 == rating_ladder_state_t::DEAD
        && state1 == rating_ladder_state_t::DEAD
        && state2 == rating_ladder_state_t::DEAD) {
        game->tactical_features2[lib2] |= po_tactical_features_mask[F_2POINT_C_ATARI];
#if 0
        cerr << "2POINT_C_ATARI " << FormatMove(lib2)
             << " CAPTURE:" << FormatMove(string[neighbor].origin)
             << " HELP:" << FormatMove(string[id].origin)
             << endl;
        PrintBoard(game);
#endif
      } else if (c_state2 == rating_ladder_state_t::DEAD || c_state2 == rating_ladder_state_t::ALIVE) {
        game->tactical_features2[lib2] |= po_tactical_features_mask[F_2POINT_ATARI];
#if 0
        cerr << "2POINT_ATARI " << FormatMove(lib2)
             << " CAPTURE:" << FormatMove(string[neighbor].origin)
             << " HELP:" << FormatMove(string[id].origin)
             << endl;
        PrintBoard(game);
#endif
      }
    }
    neighbor = string[id].neighbor[neighbor];
  }
}


/////////////////////////////////////////
//  呼吸点が3つの連に対する特徴の判定  //
/////////////////////////////////////////
static void
PoCheckFeaturesLib3( rating_context_t& ctx, const int color, const int id, position_t *update, int *update_num )
{
  game_info_t *game = ctx.game;
  const position_t *string_id = game->string_id;
  const string_t *string = game->string;
  int neighbor = string[id].neighbor[0];
  const char *board = game->board;
  int lib1, lib2, lib3;

  // 呼吸点が3つになった連の呼吸点を取り出す
  lib1 = string[id].lib[0];
  lib2 = string[id].lib[lib1];
  lib3 = string[id].lib[lib2];

  // 呼吸点の周囲が空点3つ, または呼吸点が3つ以上の自分の連に接続できるかで特徴を判定
  if (nb4_empty[Pat3(game->pat, lib1)] == 3 ||
      (board[NORTH(lib1)] == color && string_id[NORTH(lib1)] != id &&
       string[string_id[NORTH(lib1)]].libs >= 3) ||
      (board[ WEST(lib1)] == color && string_id[ WEST(lib1)] != id &&
       string[string_id[WEST(lib1)]].libs >= 3) ||
      (board[ EAST(lib1)] == color && string_id[ EAST(lib1)] != id &&
       string[string_id[EAST(lib1)]].libs >= 3) ||
      (board[SOUTH(lib1)] == color && string_id[SOUTH(lib1)] != id &&
       string[string_id[SOUTH(lib1)]].libs >= 3)) {
    game->tactical_features2[lib1] |= po_tactical_features_mask[F_3POINT_EXTENSION_SAFELY];
  } else {
    game->tactical_features2[lib1] |= po_tactical_features_mask[F_3POINT_EXTENSION];
  }

  // 呼吸点の周囲が空点3つ, または呼吸点が3つ以上の自分の連に接続できるかで特徴を判定
  if (nb4_empty[Pat3(game->pat, lib2)] == 3 ||
      (board[NORTH(lib2)] == color && string_id[NORTH(lib2)] != id &&
       string[string_id[NORTH(lib2)]].libs >= 3) ||
      (board[ WEST(lib2)] == color && string_id[ WEST(lib2)] != id &&
       string[string_id[WEST(lib2)]].libs >= 3) ||
      (board[ EAST(lib2)] == color && string_id[ EAST(lib2)] != id &&
       string[string_id[EAST(lib2)]].libs >= 3) ||
      (board[SOUTH(lib2)] == color && string_id[SOUTH(lib2)] != id &&
       string[string_id[SOUTH(lib2)]].libs >= 3)) {
    game->tactical_features2[lib2] |= po_tactical_features_mask[F_3POINT_EXTENSION_SAFELY];
  } else {
    game->tactical_features2[lib2] |= po_tactical_features_mask[F_3POINT_EXTENSION];
  }

  // 呼吸点の周囲が空点3つ, または呼吸点が3つ以上の自分の連に接続できるかで特徴を判定
  if (nb4_empty[Pat3(game->pat, lib3)] == 3 ||
      (board[NORTH(lib3)] == color && string_id[NORTH(lib3)] != id &&
       string[string_id[NORTH(lib3)]].libs >= 3) ||
      (board[ WEST(lib3)] == color && string_id[ WEST(lib3)] != id &&
       string[string_id[ WEST(lib3)]].libs >= 3) ||
      (board[ EAST(lib3)] == color && string_id[ EAST(lib3)] != id &&
       string[string_id[ EAST(lib3)]].libs >= 3) ||
      (board[SOUTH(lib3)] == color && string_id[SOUTH(lib3)] != id &&
       string[string_id[SOUTH(lib3)]].libs >= 3)) {
    game->tactical_features2[lib3] |= po_tactical_features_mask[F_3POINT_EXTENSION_SAFELY];
  } else {
    game->tactical_features2[lib3] |= po_tactical_features_mask[F_3POINT_EXTENSION];
  }

  // レートの更新対象に入れる
  update[(*update_num)++] = lib1;
  update[(*update_num)++] = lib2;
  update[(*update_num)++] = lib3;

  // 呼吸点が3つになった連の周囲の敵連を調べる
  // 1. 呼吸点が1つの敵連
  // 2. 呼吸点が2つの敵連
  // 3. 呼吸点が3つの敵連
  // それぞれに対して, 特徴を判定する
  while (neighbor != NEIGHBOR_END) {
    if (string[neighbor].libs == 1) {
      lib1 = string[neighbor].lib[0];
      update[(*update_num)++] = lib1;
      game->tactical_features1[lib1] |= po_tactical_features_mask[F_3POINT_CAPTURE];
    } else if (string[neighbor].libs == 2) {
      lib1 = string[neighbor].lib[0];
      update[(*update_num)++] = lib1;
      lib2 = string[neighbor].lib[lib1];
      update[(*update_num)++] = lib2;

      rating_ladder_state_t state1 = GetLadderState(ctx, neighbor, lib1, color);
      if (state1 == rating_ladder_state_t::DEAD) {
        game->tactical_features2[lib1] |= po_tactical_features_mask[F_3POINT_C_ATARI];
      } else if (state1 == rating_ladder_state_t::DEAD) {
        game->tactical_features2[lib1] |= po_tactical_features_mask[F_3POINT_ATARI];
      }

      rating_ladder_state_t state2 = GetLadderState(ctx, neighbor, lib2, color);
      if (state2 == rating_ladder_state_t::DEAD) {
        game->tactical_features2[lib2] |= po_tactical_features_mask[F_3POINT_C_ATARI];
      } else if (state2 == rating_ladder_state_t::ALIVE) {
        game->tactical_features2[lib2] |= po_tactical_features_mask[F_3POINT_ATARI];
      }
    }
    neighbor = string[id].neighbor[neighbor];
  }
}


//////////////////
//  特徴の判定  //
//////////////////
static void
PoCheckFeatures( rating_context_t& ctx, const int color, int previous_move, position_t *update, int *update_num )
{
  game_info_t *game = ctx.game;
  const string_t *string = game->string;
  const char *board = game->board;
  const position_t *string_id = game->string_id;
  int id;
  position_t check[3] = { 0 };
  int checked = 0;

  if (previous_move == PASS) return;

  // 直前の着手の上を確認
  if (board[NORTH(previous_move)] == color) {
    id = string_id[NORTH(previous_move)];
    if (string[id].libs == 1) {
      PoCheckFeaturesLib1(ctx, color, id, update, update_num);
    } else if (string[id].libs == 2) {
      PoCheckFeaturesLib2(ctx, color, id, update, update_num);
    } else if (string[id].libs == 3) {
      PoCheckFeaturesLib3(ctx, color, id, update, update_num);
    }
    check[checked++] = id;
  }

  // 直前の着手の左を確認
  if (board[WEST(previous_move)] == color) {
    id = string_id[WEST(previous_move)];
    if (id != check[0]) {
      if (string[id].libs == 1) {
        PoCheckFeaturesLib1(ctx, color, id, update, update_num);
      } else if (string[id].libs == 2) {
        PoCheckFeaturesLib2(ctx, color, id, update, update_num);
      } else if (string[id].libs == 3) {
        PoCheckFeaturesLib3(ctx, color, id, update, update_num);
      }
    }
    check[checked++] = id;
  }

  // 直前の着手の右を確認
  if (board[EAST(previous_move)] == color) {
    id = string_id[EAST(previous_move)];
    if (id != check[0] && id != check[1]) {
      if (string[id].libs == 1) {
        PoCheckFeaturesLib1(ctx, color, id, update, update_num);
      } else if (string[id].libs == 2) {
        PoCheckFeaturesLib2(ctx, color, id, update, update_num);
      } else if (string[id].libs == 3) {
        PoCheckFeaturesLib3(ctx, color, id, update, update_num);
      }
    }
    check[checked++] = id;
  }

  // 直前の着手の下の確認
  if (board[SOUTH(previous_move)] == color) {
    id = string_id[SOUTH(previous_move)];
    if (id != check[0] && id != check[1] && id != check[2]) {
      if (string[id].libs == 1) {
        PoCheckFeaturesLib1(ctx, color, id, update, update_num);
      } else if (string[id].libs == 2) {
        PoCheckFeaturesLib2(ctx, color, id, update, update_num);
      } else if (string[id].libs == 3) {
        PoCheckFeaturesLib3(ctx, color, id, update, update_num);
      }
    }
  }

}


////////////////////////
//  劫を解消するトリ  //
////////////////////////
static void
PoCheckCaptureAfterKo( game_info_t *game, const int color, position_t *update, int *update_num )
{
  const string_t *string = game->string;
  const char *board = game->board;
  const position_t *string_id = game->string_id;
  int other = FLIP_COLOR(color);
  int previous_move_2 = game->record[game->moves - 2].pos;
  position_t check[4] = { 0 };
  int checked = 0;

  //  上
  if (board[NORTH(previous_move_2)] == other) {
    position_t id = string_id[NORTH(previous_move_2)];
    if (string[id].libs == 1) {
      position_t lib = string[id].lib[0];
      update[(*update_num)++] = lib;
      game->tactical_features1[lib] |= po_tactical_features_mask[F_CAPTURE_AFTER_KO];
    }
    check[checked++] = id;
  }

  //  右
  if (board[EAST(previous_move_2)] == other) {
    position_t id = string_id[EAST(previous_move_2)];
    if (string[id].libs == 1 && check[0] != id) {
      position_t lib = string[id].lib[0];
      update[(*update_num)++] = lib;
      game->tactical_features1[lib] |= po_tactical_features_mask[F_CAPTURE_AFTER_KO];
    }
    check[checked++] = id;
  }

  //  下
  if (board[SOUTH(previous_move_2)] == other) {
    position_t id = string_id[SOUTH(previous_move_2)];
    if (string[id].libs == 1 && check[0] != id && check[1] != id) {
      position_t lib = string[id].lib[0];
      update[(*update_num)++] = lib;
      game->tactical_features1[lib] |= po_tactical_features_mask[F_CAPTURE_AFTER_KO];
    }
    check[checked++] = id;
  }

  //  左
  if (board[WEST(previous_move_2)] == other) {
    position_t id = string_id[WEST(previous_move_2)];
    if (string[id].libs == 1 && check[0] != id && check[1] != id && check[2] != id) {
      position_t lib = string[id].lib[0];
      update[(*update_num)++] = lib;
      game->tactical_features1[lib] |= po_tactical_features_mask[F_CAPTURE_AFTER_KO];
    }
  }
}


//////////////////
//  自己アタリ  //
//////////////////
bool
PoCheckSelfAtari( game_info_t *game, const int color, const position_t pos )
{
  const char *board = game->board;
  const string_t *string = game->string;
  const position_t *string_id = game->string_id;
  int other = FLIP_COLOR(color);
  int size = 0;
  position_t already[4] = { 0 };
  int already_num = 0;
  int lib, count = 0, libs = 0;
  position_t lib_candidate[10];
  int i;
  int id;
  bool flag;
  bool checked;

  // 上下左右が空点なら呼吸点の候補に入れる
  if (board[NORTH(pos)] == S_EMPTY) lib_candidate[libs++] = NORTH(pos);
  if (board[ WEST(pos)] == S_EMPTY) lib_candidate[libs++] =  WEST(pos);
  if (board[ EAST(pos)] == S_EMPTY) lib_candidate[libs++] =  EAST(pos);
  if (board[SOUTH(pos)] == S_EMPTY) lib_candidate[libs++] = SOUTH(pos);

  //  空点
  if (libs >= 2) return true;

  // 上を調べる
  if (board[NORTH(pos)] == color) {
    id = string_id[NORTH(pos)];
    if (string[id].libs > 2) return true;
    lib = string[id].lib[0];
    count = 0;
    while (lib != LIBERTY_END) {
      if (lib != pos) {
	checked = false;
	for (i = 0; i < libs; i++) {
	  if (lib_candidate[i] == lib) {
	    checked = true;
	    break;
	  }
	}
	if (!checked) {
	  lib_candidate[libs + count] = lib;
	  count++;
	}
      }
      lib = string[id].lib[lib];
    }
    libs += count;
    size += string[id].size;
    already[already_num++] = id;
    if (libs >= 2) return true;
  } else if (board[NORTH(pos)] == other &&
	     string[string_id[NORTH(pos)]].libs == 1) {
    return true;
  }

  // 左を調べる
  if (board[WEST(pos)] == color) {
    id = string_id[WEST(pos)];
    if (already[0] != id) {
      if (string[id].libs > 2) return true;
      lib = string[id].lib[0];
      count = 0;
      while (lib != LIBERTY_END) {
	if (lib != pos) {
	  checked = false;
	  for (i = 0; i < libs; i++) {
	    if (lib_candidate[i] == lib) {
	      checked = true;
	      break;
	    }
	  }
	  if (!checked) {
	    lib_candidate[libs + count] = lib;
	    count++;
	  }
	}
	lib = string[id].lib[lib];
      }
      libs += count;
      size += string[id].size;
      already[already_num++] = id;
      if (libs >= 2) return true;
    }
  } else if (board[WEST(pos)] == other &&
	     string[string_id[WEST(pos)]].libs == 1) {
    return true;
  }

  // 右を調べる
  if (board[EAST(pos)] == color) {
    id = string_id[EAST(pos)];
    if (already[0] != id && already[1] != id) {
      if (string[id].libs > 2) return true;
      lib = string[id].lib[0];
      count = 0;
      while (lib != LIBERTY_END) {
	if (lib != pos) {
	  checked = false;
	  for (i = 0; i < libs; i++) {
	    if (lib_candidate[i] == lib) {
	      checked = true;
	      break;
	    }
	  }
	  if (!checked) {
	    lib_candidate[libs + count] = lib;
	    count++;
	  }
	}
	lib = string[id].lib[lib];
      }
      libs += count;
      size += string[id].size;
      already[already_num++] = id;
      if (libs >= 2) return true;
    }
  } else if (board[EAST(pos)] == other &&
	     string[string_id[EAST(pos)]].libs == 1) {
    return true;
  }


  // 下を調べる
  if (board[SOUTH(pos)] == color) {
    id = string_id[SOUTH(pos)];
    if (already[0] != id && already[1] != id && already[2] != id) {
      if (string[id].libs > 2) return true;
      lib = string[id].lib[0];
      count = 0;
      while (lib != LIBERTY_END) {
	if (lib != pos) {
	  checked = false;
	  for (i = 0; i < libs; i++) {
	    if (lib_candidate[i] == lib) {
	      checked = true;
	      break;
	    }
	  }
	  if (!checked) {
	    lib_candidate[libs + count] = lib;
	    count++;
	  }
	}
	lib = string[id].lib[lib];
      }
      libs += count;
      size += string[id].size;
      already[already_num++] = id;
      if (libs >= 2) return true;
    }
  } else if (board[SOUTH(pos)] == other &&
	     string[string_id[SOUTH(pos)]].libs == 1) {
    return true;
  }

  if (size == 2) {
    // ooo#
    // o+@#
    // o@+#
    // ####
    int pos22x = -1;
    int pos22y = -1;
    int x = X(pos);
    int y = Y(pos);
    if (x == board_start || x == board_start + 1)
      pos22x = board_start + 1;
    else if (x == board_end - 1 || x == board_end)
      pos22x = board_end - 1;
    if (y == board_start || y == board_start + 1)
      pos22y = board_start + 1;
    else if (y == board_end - 1 || y == board_end)
      pos22y = board_end - 1;
    if (pos22x > 0 && pos22y > 0) {
      int pos22 = POS(pos22x, pos22y);
      int md2 = MD2(game->pat, pos22);
      if (seki_22_set[color - 1].count(md2) > 0) {
        return false;
      }
    }
  }

  // 自己アタリになる連の大きさが2以下,
  // または大きさが5以下でナカデの形になる場合は
  // 打っても良いものとする
  if (size == 0) {
    game->tactical_features2[pos] |= po_tactical_features_mask[F_SELF_ATARI_SMALL];
    flag = true;
  } else if (size < 2) {
    game->tactical_features2[pos] |= po_tactical_features_mask[F_SELF_ATARI_SMALL];
    flag = true;
  } else if (size < 5) {
    if (IsNakadeSelfAtari(game, pos, color)) {
      game->tactical_features2[pos] |= po_tactical_features_mask[F_SELF_ATARI_NAKADE];
      flag = true;
    } else {
      game->tactical_features2[pos] |= po_tactical_features_mask[F_SELF_ATARI_LARGE];
      flag = false;
    }
  } else {
    game->tactical_features2[pos] |= po_tactical_features_mask[F_SELF_ATARI_LARGE];
    flag = false;
  }

  return flag;
}


static void
PoCheckCaptureAndAtari( rating_context_t& ctx, const int color, const position_t pos, position_t other_pos )
{
#define IsSafe(id, p) (board[p] == other && id != string_id[p] && string[string_id[p]].libs > 2)
  game_info_t *game = ctx.game;
  const char *board = game->board;
  const string_t *string = game->string;
  const position_t *string_id = game->string_id;
  const int other = FLIP_COLOR(color);

  if (board[other_pos] == other) {
    int id = string_id[other_pos];
    int libs = string[id].libs;
    if (libs == 1) {
      if (string[id].size == 1) {
        game->tactical_features1[pos] |= po_tactical_features_mask[F_CAPTURE1];
      } else if (string[id].size == 2) {
        game->tactical_features1[pos] |= po_tactical_features_mask[F_CAPTURE2];
      } else if (string[id].size == 3) {
        game->tactical_features1[pos] |= po_tactical_features_mask[F_CAPTURE3];
      } else {
        game->tactical_features1[pos] |= po_tactical_features_mask[F_CAPTURE4];
      }

      if (string[id].size > 2) {
        /*
        bool escapable = false;
        int neighbor = string[id].neighbor[0];
        while (neighbor != NEIGHBOR_END) {
          if (string[neighbor].libs == 1) {
            escapable = true;
            break;
          }
          neighbor = string[id].neighbor[neighbor];
        }
        if (!escapable
            && (IsSafe(id, NORTH(pos))
                || IsSafe(id, WEST(pos))
                || IsSafe(id, EAST(pos))
                || IsSafe(id, SOUTH(pos)))) {
          escapable = true;
        }
        */
        int max_size = string[id].size;
        if (!IsLadderCaptured(0, ctx.search_game, string[id].origin, FLIP_COLOR(color), max_size)) {
#if 0
          cerr << "NO ESCAPABLE " << FormatMove(pos) << " " << FormatMove(other_pos) << endl;
          PrintBoard(game);
#endif
        } else {
          game->tactical_features1[pos] |= po_tactical_features_mask[F_CAPTURE_ESCAPABLE];
#if 0
          cerr << "ESCAPABLE " << FormatMove(pos) << " " << FormatMove(other_pos) << endl;
          PrintBoard(game);
#endif
        }
        /*
        if (escapable) {
          //cerr << "ESCAPABLE " << FormatMove(pos) << " " << FormatMove(other_pos) << endl;
          //PrintBoard(game);
          game->tactical_features1[pos] |= po_tactical_features_mask[F_CAPTURE_ESCAPABLE];
        }
        */
      }
    } else if (libs == 2) {
      //bool capturable = IsCapturableAtariForSimulation(game, pos, color, id);
      /*
      if (capturable) {
        game->tactical_features2[pos] |= po_tactical_features_mask[F_C_ATARI];
      } else {
        game->tactical_features2[pos] |= po_tactical_features_mask[F_ATARI];
      }
      */
      bool legal = true;
      //if (ctx.string_captured[id] == rating_ladder_state_t::UNCHECKED) {
      if (IsLegalForSearch(ctx.search_game, pos, color)) {
      } else {
        legal = false;
      }

      if (!legal) {
        //SKIP
      } else {
        game->tactical_features2[pos] |= po_tactical_features_mask[F_ATARI];
      }
    }
  }
#undef IsSafe
}


//////////////////
//  トリの判定  //
//////////////////
static void
PoCheckCaptureAndAtari( rating_context_t& ctx, const int color, const position_t pos )
{
  game_info_t *game = ctx.game;
  const char *board = game->board;

  // 上を調べる
  // 1. 敵の石
  // 2. 呼吸点が1つ
  PoCheckCaptureAndAtari(ctx, color, pos, NORTH(pos));

  //  左を調べる
  // 1. 敵の石
  // 2. 呼吸点が1つ
  PoCheckCaptureAndAtari(ctx, color, pos, WEST(pos));

  //  右を調べる
  // 1. 敵の石
  // 2. 呼吸点が1つ
  PoCheckCaptureAndAtari(ctx, color, pos, EAST(pos));

  //  下を調べる
  // 1. 敵の石
  // 2. 呼吸点が1つ
  PoCheckCaptureAndAtari(ctx, color, pos, SOUTH(pos));
}


void
AnalyzePoRating( rating_context_t& ctx, int color, double rate[] )
{
  game_info_t *game = ctx.game;
  int moves = game->moves;
  int pm1 = PASS;
  position_t update_pos[BOARD_MAX];
  int update_num = 0;
  const position_t *string_id = game->string_id;
  const string_t *string = game->string;
  const char *board = game->board;

  for (int i = 0; i < pure_board_max; i++) {
    int pos = onboard_pos[i];
    game->tactical_features1[pos] = 0;
    game->tactical_features2[pos] = 0;
  }

  ctx.clear();

  pm1 = game->record[moves - 1].pos;

  PoCheckFeatures(ctx, color, pm1, update_pos, &update_num);
  if (game->ko_move == moves - 2) {
    PoCheckCaptureAfterKo(game, color, update_pos, &update_num);
  }

  for (int id = 0; id < MAX_STRING; id++) {
    if (!string[id].flag || string[id].color != color)
      continue;
    update_num = 0;
    if (string[id].libs == 1) {
      PoCheckFeaturesLib1(ctx, color, id, update_pos, &update_num);
    } else if (string[id].libs == 2) {
      PoCheckFeaturesLib2(ctx, color, id, update_pos, &update_num);
    } else if (string[id].libs == 3) {
      PoCheckFeaturesLib3(ctx, color, id, update_pos, &update_num);
    }
  }

  for (int i = 0; i < pure_board_max; i++) {
    int pos = onboard_pos[i];

    if (!IsLegal(game, pos, color)) {
      rate[i] = 0;
      continue;
    }

    PoCheckSelfAtari(game, color, pos);
    PoCheckCaptureAndAtari(ctx, color, pos);

#if 0
    double gamma = 1.0;

    if (pm1 != PASS) {
      if (DIS(pos, pm1) == 2) {
        gamma *= po_previous_distance[0];
      } else if (DIS(pos, pm1) == 3) {
        gamma *= po_previous_distance[1];
      } else if (DIS(pos, pm1) == 4) {
        gamma *= po_previous_distance[2];
      }
    }

    rate[i] = GetPoGamma(game, color, pos, gamma);
#endif
  }
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

}
