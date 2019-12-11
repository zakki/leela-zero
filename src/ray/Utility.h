#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <chrono>
#include <memory>


////////////
//  関数  //
////////////
typedef std::chrono::high_resolution_clock ray_clock;

// 消費時間の算出
inline double GetSpendTime(const ray_clock::time_point& start_time) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(ray_clock::now() - start_time).count() / 1000.0;
}

//  データ読み込み(float)
void InputTxtFLT( const char *filename, float *ap, const int array_size );

//  データ読み込み(double)
void InputTxtDBL( const char *filename, double *ap, const int array_size );

#if !defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#if __cplusplus < 201402L
template<typename T, typename ...Args>
std::unique_ptr<T> make_unique(Args&& ...args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif
#endif

template<typename T>
T inline clip(T n, T lower, T upper) {
  return std::max(lower, std::min(n, upper));
}

#endif
