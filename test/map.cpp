#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>

#include "zensim/container/RBTreeMap.hpp"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/core.h"
// #include "zensim/omp/execution/ExecutionPolicy.hpp"

int main() {
  using namespace zs;
  {
    std::map<std::string, float> m1{};
    RBTreeMap<std::string, float> m2{};
    std::vector<std::pair<std::string, float>> v1{};
    const int N = 1000000;
    const int M = 10000;
    zs::CppTimer timer;
    std::srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N; ++i) {
      v1.emplace_back(std::make_pair(std::to_string(std::rand() / M), (float)std::rand() / M));
      // fmt::print("{}-{}\n", v1[i].first, v1[i].second);
    }
    // insert check
    timer.tick();
    for (int i = 0; i < N; ++i) m1[v1[i].first] = v1[i].second;
    timer.tock(fmt::format(" std map insert"));
    timer.tick();
    for (int i = 0; i < N; ++i) m2[v1[i].first] = v1[i].second;
    timer.tock(fmt::format(" zs map insert"));
    auto it2 = m2.begin();
    for (auto it1 = m1.begin(); it1 != m1.end(); ++it1, ++it2) {
      if (it2 == m2.end()) throw std::runtime_error("iterate error");
      if (it1->first != it2->first || std::fabs(it1->second - it2->second) > 1e-5)
        throw std::runtime_error("iterate error");
    }
    // find check
    std::vector<std::string> v3{};
    for (int i = 0; i < N / 3; ++i) v3.emplace_back(v1[std::rand() % N].first);
    for (int i = 0; i < N / 3 * 2; ++i) v3.emplace_back(std::to_string(std::rand() / M * N));
    // std::random_shuffle(v3.begin(), v3.end()); // deprecated
    std::mt19937 rng(std::time(nullptr));
    std::shuffle(v3.begin(), v3.end(), rng);
    std::vector<int> f1{};
    std::vector<int> f2{};
    timer.tick();
    for (int i = 0; i < v3.size(); ++i) f1.emplace_back(m1.count(v3[i]));
    timer.tock(fmt::format(" std map count"));
    timer.tick();
    for (int i = 0; i < v3.size(); ++i) f2.emplace_back(m2.count(v3[i]));
    timer.tock(fmt::format(" zs map count"));
    for (int i = 0; i < v3.size(); ++i) {
      if (f1[i] != f2[i]) {
        throw std::runtime_error("find error");
      }
    }
    // delete check
    std::set<int> s1{};
    std::vector<int> v2{};
    for (int i = 0; i < N; ++i) {
      int idx = std::rand() % N;
      if (s1.count(idx) == 0) {
        s1.insert(idx);
        v2.emplace_back(idx);
      }
    }
    if (s1.count(0) == 0) {
      s1.insert(0);
      v2.emplace_back(0);
    }
    if (s1.count(N - 1) == 0) {
      s1.insert(N - 1);
      v2.emplace_back(N - 1);
    }
    timer.tick();
    for (int i = 0; i < v2.size(); ++i) m1.erase(v1[v2[i]].first);
    timer.tock(fmt::format(" std map erase"));
    timer.tick();
    for (int i = 0; i < v2.size(); ++i) m2.erase(v1[v2[i]].first);
    timer.tock(fmt::format(" zs map erase"));
    if (m1.size() != m2.size()) throw std::runtime_error("delete error");
    it2 = m2.begin();
    for (auto it1 = m1.begin(); it1 != m1.end(); ++it1, ++it2) {
      if (it2 == m2.end()) throw std::runtime_error("iterate error");
      if (it1->first != it2->first || std::fabs(it1->second - it2->second) > 1e-5)
        throw std::runtime_error("iterate error");
    }

    // std::map<int, float> m1{};
    // RBTreeMap<int, float> m2{};
    // std::vector<std::pair<int, float>> v1{};
    // const int N = 1000000;
    // const int M = 10000;
    // zs::CppTimer timer;
    // std::srand(static_cast<unsigned int>(time(0)));
    // for (int i = 0; i < N; ++i) {
    //   v1.emplace_back(std::make_pair(std::rand()/M, (float)std::rand()/M));
    //   // fmt::print("{}-{}\n", v1[i].first, v1[i].second);
    // }
    // // insert check
    // timer.tick();
    // for (int i = 0; i < N; ++i)
    //   m1[v1[i].first] = v1[i].second;
    // timer.tock(fmt::format(" std map insert"));
    // timer.tick();
    // for (int i = 0; i < N; ++i)
    //   m2[v1[i].first] = v1[i].second;
    // timer.tock(fmt::format(" zs map insert"));
    // auto it2 = m2.begin();
    // for (auto it1= m1.begin(); it1 != m1.end(); ++it1, ++it2) {
    //   if (it2 == m2.end()) throw std::runtime_error("iterate error");
    //   if (it1->first != it2->first || std::fabs(it1->second - it2->second) > 1e-5)
    //     throw std::runtime_error("iterate error");
    // }
    // // find check
    // std::vector<int> v3{};
    // for (int i = 0; i < N/3; ++i)
    //   v3.emplace_back(v1[std::rand()%N].first);
    // for (int i = 0; i < N/3*2; ++i)
    //   v3.emplace_back(std::rand()/M*N);
    // std::random_shuffle(v3.begin(), v3.end());
    // std::vector<int> f1{};
    // std::vector<int> f2{};
    // timer.tick();
    // for (int i = 0; i < v3.size(); ++i)
    //   f1.emplace_back(m1.count(v3[i]));
    // timer.tock(fmt::format(" std map count"));
    // timer.tick();
    // for (int i = 0; i < v3.size(); ++i)
    //   f2.emplace_back(m2.count(v3[i]));
    // timer.tock(fmt::format(" zs map count"));
    // for (int i = 0; i < v3.size(); ++i) {
    //   if (f1[i] != f2[i]) {
    //     throw std::runtime_error("find error");
    //   }
    // }
    // // delete check
    // std::set<int> s1{};
    // std::vector<int> v2{};
    // for (int i = 0; i < N; ++i) {
    //   int idx = std::rand()%N;
    //   if (s1.count(idx) == 0) {
    //     s1.insert(idx);
    //     v2.emplace_back(idx);
    //   }
    // }
    // timer.tick();
    // for (int i = 0; i < v2.size(); ++i)
    //   m1.erase(v1[v2[i]].first);
    // timer.tock(fmt::format(" std map erase"));
    // timer.tick();
    // for (int i = 0; i < v2.size(); ++i)
    //   m2.erase(v1[v2[i]].first);
    // timer.tock(fmt::format(" zs map erase"));
    // if (m1.size() != m2.size()) throw std::runtime_error("delete error");
    // it2 = m2.begin();
    // for (auto it1= m1.begin(); it1 != m1.end(); ++it1, ++it2) {
    //   if (it2 == m2.end()) throw std::runtime_error("iterate error");
    //   if (it1->first != it2->first || std::fabs(it1->second - it2->second) > 1e-5)
    //     throw std::runtime_error("iterate error");
    // }
  }
  return 0;
}