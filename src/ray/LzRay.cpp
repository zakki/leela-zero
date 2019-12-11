#include <cstring>
#include <cstdio>
#include <iostream>
#include <memory>

#if defined (_WIN32)
#include <windows.h>
#endif

#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
//#include <boost/python/numpy.hpp>

#include "GoBoard.h"
//#include "Gtp.h"
#include "Message.h"
//#include "OpeningBook.h"
//#include "PatternHash.h"
#include "Rating2.h"
#include "Semeai.h"
//#include "Train.h"
//#include "UctRating.h"
//#include "UctSearch.h"
#include "ZobristHash.h"


//namespace py = boost::python;
//namespace numpy = boost::python::numpy;


void ReadLzPlane(const std::vector<float>& planes, int chanel, int color, game_info_t* game)
{
  for (auto n = size_t{ 0 }; n < pure_board_max; n++) {
    //auto c = planes[n * (18 + num_features) + chanel];
    auto c = planes[pure_board_max * chanel + n];
    if ((c > 0) && game->board[onboard_pos[n]] == S_EMPTY) {
      if (!IsLegal(game, onboard_pos[n], color)) {
        std::cerr << "Illegal move " << FormatMove(onboard_pos[n]) << std::endl;
        PrintBoard(game);
        throw std::runtime_error("Illegal move");
      }
      PutStone(game, onboard_pos[n], color);
    }
  }
}

void WritePlanes(std::vector<float>& planes, int color)
{
  auto game = std::unique_ptr<game_info_t>(AllocateGame());
  ClearBoard(game.get());

  for (int i = 7; i >= 0; i--) {
    int moves0 = game->moves;
    ReadLzPlane(planes, (color == S_BLACK ? i : 8 + i), S_BLACK, game.get());
    ReadLzPlane(planes,  (color == S_BLACK ? 8 + i : i), S_WHITE, game.get());
    if (game->moves > 1 && game->moves == moves0) {
      PutStone(game.get(), PASS, FLIP_COLOR(game->record[game->moves - 1].color));
    }
    // if (win == 100)
    //PrintBoard(game.get());
  }

  WritePlanes2(planes.data() + pure_board_max * 18, game.get(), color, 0);
}


void collect_features(std::vector<float>& planes, bool color) {
  size_t N = planes.size();
  if (N != 19 * 19 * (18 + num_features))
    throw std::runtime_error("planes must be 19 * 19 * N " + std::to_string(N));
  if (color != 0 && color != 1)
    throw std::runtime_error("illegal color");
  WritePlanes(planes, color ? S_BLACK : S_WHITE);
}


void init_ray() {

#if 0
  char program_path[1024];
  int last;

  // 実行ファイルのあるディレクトリのパスを抽出
#if defined (_WIN32)
  HMODULE hModule = GetModuleHandle(NULL);
  GetModuleFileNameA(hModule, program_path, 1024);
#else
  strcpy(program_path, argv[0]);
#endif
  last = (int)strlen(program_path);
  while (last--){
#if defined (_WIN32)
    if (program_path[last] == '\\' || program_path[last] == '/') {
      program_path[last] = '\0';
      break;
    }
#else
    if (program_path[last] == '/') {
      program_path[last] = '\0';
      break;
    }
#endif
  }

  // 各種パスの設定
  /*
#if defined (_WIN32)
  sprintf_s(uct_params_path, 1024, "%s\\uct_params", program_path);
  sprintf_s(po_params_path, 1024, "%s\\sim_params", program_path);
#else
  snprintf(uct_params_path, 1024, "%s/uct_params", program_path);
  snprintf(po_params_path, 1024, "%s/sim_params", program_path);
#endif
  */
#endif

  // 各種初期化
  InitializeConst();
  //InitializeRating();
  //InitializeUctRating();
  InitializeHash();
  InitializeUctHash();
  rating_v2::SetNeighbor();

  //py::def("collect_features", collect_features);
}
