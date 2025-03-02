#if 0
#  include "zensim/musa/Musa.h"

int main() {
  using namespace zs;
  (void)Musa::instance();
  return 0;
}

#else
#  include <musa.h>
#  include <musa_runtime.h>

#  include <iostream>

#  include "zensim/execution/Atomics.hpp"
#  include "zensim/execution/Intrinsics.hpp"
#  include "zensim/musa/Musa.h"
#  include "zensim/musa/execution/ExecutionPolicy.muh"
#  include "zensim/types/Mask.hpp"

template <size_t... Is> void tttt(std::index_sequence<Is...>) {
  ((void)(std::cout << Is << ','), ...);
}

template <typename T> constexpr T add(T a, T b) { return a + b; }

template <typename Fn> __global__ void test(Fn fn) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("[%d]: %f, %f\n", idx, add((float)idx, ::sqrt(idx * 1.2f)), fn(idx));
}

template <typename Vn> __global__ void test_access(Vn vn) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("[%d]: %d\n", idx, vn[idx]);
}

int main() {
  if constexpr (true) {
    tttt(std::make_index_sequence<5>{});
    std::cout << "succeed" << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
  using namespace zs;
#  if 0
  MUresult result = muInit(0);
  if (result != MUSA_SUCCESS) {
    std::cout << "[muInit] MUSA error: " << result << '\n';
    return -1;
  }

  int numTotalDevice = 0;
  result = muDeviceGetCount(&numTotalDevice);
  if (result != MUSA_SUCCESS) {
    std::cout << "[muDeviceGetCount] MUSA error: " << result << '\n';
    return -1;
  } else if (numTotalDevice == 0) {
    std::cout << "No musa device detected" << '\n';
    return -1;
  }

  for (int i = 0; i < numTotalDevice; ++i) {
    MUdevice dev = 0;
    MUcontext ctx;
    muDeviceGet(&dev, i);

    int major, minor;
    result = muDeviceGetAttribute(&minor, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (result != MUSA_SUCCESS) {
      std::cout << "[muDeviceGetAttribute: minor] MUSA error: " << result << '\n';
      return -1;
    }
    result = muDeviceGetAttribute(&major, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    if (result != MUSA_SUCCESS) {
      std::cout << "[muDeviceGetAttribute: major] MUSA error: " << result << '\n';
      return -1;
    }
    unsigned int const compute_capability = major * 10 + minor;
    std::cout << compute_capability << ' ';
  }
#  endif

  // test<<<1, 32>>>([a = 1] __device__(int x) -> float { return x * x; });
  // musaDeviceSynchronize();

  auto pol = zs::MusaExecutionPolicy{};
  constexpr auto space = zs::execspace_e::musa;
  auto vs = zs::Vector<int>{10, zs::memsrc_e::um, 0};
  for (int i = 0; i < vs.size(); ++i) vs[i] = i * i;
  puts("access on device");
  test_access<<<1, vs.size()>>>(view<space>(vs));
  musaDeviceSynchronize();

  puts("access on host");
  // vs = vs.clone({memsrc_e::host, -1});
  for (int i = 0; i < vs.size(); ++i) printf("on host: %d: %d\n", i, vs[i]);
  pol(range(100), [] __device__(int i) {});
  pol(range(100), [vs = view<space>(vs)] __device__(int i) mutable { vs[i] = i * i; });
  pol(enumerate(vs),
      [] __device__(int i, int n) { printf("on device (through policy): [%d]: %d\n", i, n); });

  bit_mask<66> bm;
  bm.setOn(16);
  bm.setOn(2);
  bm.setOn(1);
  bm.setOn(0);
  bm.setOn(32);
  bm.setOn(64);
  fmt::print("num ones: {}\n", bm.countOn(seq_c));
  printf("abs of -1.3: %f\n", abs(-1.3f, seq_c));
  {
    zs::Vector<int> vs{1, zs::memsrc_e::device, -1};
    vs.reset(0);
    pol(range(1), [vs = view<space>(vs)] __device__(int i) mutable {
      bit_mask<66> bm;
      bm.setOn(16);
      bm.setOn(2);
      bm.setOn(1);
      bm.setOn(0);
      bm.setOn(32);
      bm.setOn(64);
      printf("%d-th bit num ons: %d\n", 10, bm.countOffset(16));

      printf("abs of -1.3: %f\n", abs(-1.3f));
#  ifdef __MUSA_ARCH__
      printf("musa arch: %d\n", (int)__MUSA_ARCH__);
#  endif
      // atomic_add(exec_musa, &vs[i], 1.);
      // atomic_inc(exec_musa, &vs[i]);
      atomic_xor(exec_musa, &vs[i], 0x33);
      // atomicAdd(&vs[i], 1.);
    });
    vs = vs.clone({memsrc_e::host, -1});
    fmt::print("result: {}\n", vs[0]);
  }

  return 0;
}

#endif
