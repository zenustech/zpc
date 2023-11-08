# Zpc框架介绍

ZPC（Zenus Parallel Compute）是一个基于cmake构建的面向异构计算（当前仅支持x64架构）的跨平台（目前windows/linux）c++（数据）并行编程框架，核心目标是支持物理仿真领域相关应用的高效研发（Zenus所稳定维护的Zensim物理仿真库即是基于ZPC而构建）。ZPC在使用接口方面看齐c++标准库的泛型编程风格，并对一系列并行编程中常见的计算模式（如for_each、reduce、scan、radix_sort等）进行封装，提供了各种计算后端（如openmp、cuda）下的高性能实现（采用多编译器编译模型，即host端c++编译器msvc、clang、gcc等配合nvcc、hip等device compiler共同构建；未来会过渡到SYCL单编译器编译模型）。开发者在开发并行算法时能够直接基于ZPC所提供的有多后端支持的各类并行模式以及常见的内建函数（如atomic operations、thread_fence、clz等）进行编程。ZPC的开发借鉴了[kokkos](https://github.com/kokkos/kokkos)、[raja](https://github.com/LLNL/RAJA)、[sycl](https://www.khronos.org/sycl/)等框架或语言在数据并行方面的设计概念与具体实现。为满足物理仿真的实际需求，ZPC还同时开发维护着一系列与空间相关的数据结构（空间哈希、BVH、自适应网格等）以支持高效率的空间操作。

## 项目构建和使用

ZPC的构建与使用都是基于cmake工具链。

### 构建选项

ZPC面向开发者提供以下构建选项：

* 各类计算后端
  * ZS_ENABLE_OPENMP：建议OpenMP4.5+版本
  * ZS_ENABLE_CUDA：需要cuda11.3+版本
* 功能模块
  * ZS_ENABLE_OPENVDB：vdb相关文件IO，vdb数据结构转换等功能
* 其他构建选项
  * ZS_BUILD_SHARED_LIBS：是否以shared library方式构建zpc、zpctool等cmake target

### 使用途径

* 作为子模块（submodule）与父项目一同从源码构建
* 通过cmake安装并通过find_package(zensim)引入项目（需要进一步测试）

## 数据管理和维护

​		数据的访问效率是决定程序整体性能的关键一环，这对于异构架构系统带来了额外挑战。以CPU/GPU系统为例，两种计算硬件所使用的内存以及缓存结构有着显著区别，而且由于硬件执行逻辑的差异，两者对于数据的访问模式（access pattern）通常也不相同；另一方面，数据存储的位置在物理上是独立的，通常情况下这些存储部件相互间的通信与数据传输需要经过带宽非常有限的PCIe4.0甚至3.0。为了达到更高的执行性能，开发者往往需要谨慎处理繁琐的编程细节，包括手动在不同存储空间间传输数据，设计合适的数据结构以及算法等，以便更有效率地利用计算硬件。

​		ZPC在管理和维护数据时从以下两方面应来处理上述问题：

1. 对ZPC构建时启用的所有计算后端所涵盖的存储空间（memory space）进行分类（比如cuda后端所支持的device memory和unified memory），并封装常用的存储操作（比如allocate, deallocate, memset, memcpy等）
2. ZPC提供基于**[结构结点]()**的数据结构快速组装和定义功能，以及运行时设置域大小和通道数量的特性支持。方便开发者快速做原型设计

​		此外，为了便于用户更直接地开发物理仿真算法，ZPC自身还提供*Vector*、*SoAVector*（TBD）、*TileVector*（即AoSoA Vector）、*HashTable*等基础数据结构，以及基于此构建的*IndexBuckets*（用于近邻查询）、*Linear BVH*（碰撞检测、光线追踪）、*Sparse Grid*、*Particles*、*Adaptive Grid*（TBD）等一系列空间数据结构和仿真数据结构，还包含*Sparse Matrix*等线性系统解算所需的结构。

### 存储空间（memory space）

目前ZPC支持x64架构下的CPU/GPU异构编程，这里

```cpp
/// FILE： zpc/include/zensim/memory/MemoryResource.h

// NOTE: 目前的 device 和 um 默认为cuda后端的实现
// 存储空间枚举类型
enum memsrc_e : unsigned char { host, device, um };

// NOTE: template <auto N> using wrapv = std::constant_integral<decltype(N), N>;
// 存储空间Tag类型及变量，用于tag dispatch
using host_mem_tag = 		wrapv<memsrc_e::host>;
using device_mem_tag = 		wrapv<memsrc_e::device>;
using um_mem_tag = 			wrapv<memsrc_e::um>;
constexpr host_mem_tag		mem_host{};
constexpr device_mem_tag	mem_device{};
constexpr um_mem_tag		mem_um{};

using mem_tags = variant<host_mem_tag, device_mem_tag, um_mem_tag>;

//
using ProcID = char;	// processor id (or compute device index). -1 usually means cpu
struct MemoryLocation {
  memsrc_e space;
  ProcID devid;
};
```

### 重要概念和使用场景

​		为了编写可拓展的（可作用于各类range）算法，c++标准库采用iterator来帮助泛型编程（generic programming）。

* 迭代器（iterator）

  ​	iterator是开发者访问数据结构内部数据的通用接口，通常需要提供*dereference*、*advance*和*compare*三个方面的实现。C++17根据advance的行为特点分出了[五类iterator](https://en.cppreference.com/w/cpp/iterator)。为了能更高效地并行访问容器内的数据，开发者通常期望在O(1)时间内能通过迭代器随机访问range内的任意元素，这类iterator的iterator_category是random_access_iterator_tag类型。

  ​	在ZPC框架内用户实现可随机访问的iterator时相较于普通的iterator还要额外提供advance和distance_to两个接口的实现，然后通过*IteratorInterface*来自动补充其余标准库关于iterator接口的实现：

  ```cpp
  /// FILE: zpc/include/zensim/types/Iterator.h
  /// ZPC框架里采用CRTP的方式来简化iterator的实现（即用户只需定义iterator的少量成员函数，无需提供其余要求的接口）
  template <typename Derived> struct IteratorInterface {
    ...
  };
  ```

  当然并非所有ZPC框架内的迭代器都要求是random_access_iterator_tag类别，比如在MPM的tranfer环节用到的[arena range](https://github.com/zenustech/zpc/blob/730a7b4fd2a9ec5bf07c6e288d9513b619682e44/include/zensim/simulation/Utils.hpp#L59)是基于forward_iterator_tag类别的[ndrange](https://github.com/zenustech/zpc/blob/730a7b4fd2a9ec5bf07c6e288d9513b619682e44/include/zensim/types/Iterator.h#L443)而实现；其他语言中常见的[zip](https://github.com/zenustech/zpc/blob/730a7b4fd2a9ec5bf07c6e288d9513b619682e44/include/zensim/types/Iterator.h#L461)、[enumerate](https://github.com/zenustech/zpc/blob/730a7b4fd2a9ec5bf07c6e288d9513b619682e44/include/zensim/types/Iterator.h#L534)等range对应的多元组iterator类别在ZPC框架内则是依据内部各iterator类别共同决定。

  

* 存储分配器（memory allocator）

  ZPC里的存储分配器（ZSPmrAllocator）基于c++17的polymorphic memory resource（PMR）去搭建，并额外内嵌MemoryLocation信息。

  ```cpp
  /// FILE: zpc/include/zensim/memory/Allocator.h
  /// Reference: https://en.cppreference.com/w/cpp/named_req/Allocator
  template <typename MemTag = host_mem_tag>
  struct raw_memory_resource : std::memory_resource {...};  // Singleton
  struct default_memory_resource : std::memory_resource {...};
  struct advisor_memory_resource : std::memory_resource {...};
  /// FILE: zpc/include/zensim/resource/Resource.h
  // 不同于std::polymorphic_allocator<T>，ZSPmrAllocator<T>在move、copy、swap时会propagate
  template <typename T>
  struct ZSPmrAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    // 以下三种属性有别于std::polymorphic_allocator<T>
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    ……
    
    std::shared_ptr<std::memory_resource> res{};	// holds an allocator (owning) or an upstream memory resource (non-owning)
    MemoryLocation memLoc{}; // describe the memory space and the device index where this allocator operates
  };
  ```

  

* 范围（range）

  range简单的定义是一对由iterator所指定的范围，它可分为以下两种类别

  * 容器（container）：需要维护数据存储，RAII。一般会提供迭代器和存储分配器
  * 视图（view）：轻量级的访问容器的视图对象，对于const和non-const的容器的视图有差异

* 存储布局（memory layout）
  * aos
  * soa
  * aosoa

### 跨内存空间的数据迁移

同步性

显式迁移

​		使用数据结构统一提供的clone(...)函数

隐式迁移

​		基于状态自动机

### 结构结点（structural node）

​		下一步准备采用boost-hana模板元编程库来替换现有实现

### 下一步重点研发方向

* 对以句柄（handle）表达的存储对象进行封装（类比SYCL的buffer，包括accessor），与目前ZPC里基于地址的存储对象进行接口的统一
* 对虚拟存储典型操作的封装（reserve、allocation、map、unmap、free、release等）
* 对texture memory、constant memory、file memory等特殊存储空间的支持完善

## 计算执行策略（execution policy）

随着计算核心的频率越来越逼近物理极限（受限于有限的散热能力无法处理大量的发热）时，增加核数成为了计算硬件发展更强大计算能力的重要手段，即并行计算能力。ZPC参考了kokkos中的*ExecutionPolicy*以及SYCL中的*Handler*等概念抽象与实现，通过“执行策略”类型给开发者提供统一的并行编程接口（uniform interface）并隐藏实际并行计算时后端的运行细节，此外还对常用的并行计算模式进行封装以助于更快速的开发。

### 执行空间（execution space）

```cpp
/// FILE: zpc/include/zensim/execution/ExecutionPolicy.hpp

// 执行空间枚举类型
enum execspace_e : unsigned char { host, openmp, cuda, hip };	// hip not impled yet

// 执行空间Tag类型及变量，用于tag dispatch
using host_exec_tag = 		wrapv<execspace_e::host>;
using omp_exec_tag = 		wrapv<execspace_e::openmp>;
using cuda_exec_tag = 		wrapv<execspace_e::cuda>;
using hip_exec_tag = 		wrapv<execspace_e::hip>;
constexpr host_exec_tag		exec_seq{};
constexpr omp_exec_tag		exec_omp{};
constexpr cuda_exec_tag		exec_cuda{};
constexpr hip_exec_tag		exec_hip{};

using exec_tags = variant<host_exec_tag, omp_exec_tag, cuda_exec_tag, hip_exec_tag>;

//
template <typename ExecTag>
constexpr auto par_exec(ExecTag) noexcept { 
  return ...; // return an execution policy object corresponding to the required execution backend (e.g. cuda)
}
```

​		所有cuda后端的实现均位于**zpc/include/zensim/cuda/**文件夹内，所有openmp后端的实现则位于**zpc/include/zensim/omp/**文件夹内。

### 执行策略（execution policy）调用形式

​		在介绍执行策略调用形式前需要完成配置并准备好待执行的函数体（functor）：

```cpp
/// macros
/// FILE: zpc/include/zensim/TypeAlias.hpp
#define ZS_FUNCTION ...
#define ZS_LAMBDA ...
/// execution policy
auto execPol = par_exec(wrapv<execspace_e::cuda>{});
/// configure policy
execPol.device(1).sync(true).profile(true)
/// execution body （返回值要求是void，参数要求值传递（比如iterator））
ZS_FUNCTION void execBody0(Iterators... iterators) {
  ...;
}
auto execBody1 = [...] ZS_LAMBDA(...) -> void { ...; };
```

​		基于ZPC的并行计算调用形式接近std<algorithm>里的形式，

1. parallel_pattern(policy, args... (iterators, etc.))

```cpp
exclusive_scan(execPol, std::begin(rangeIn), std::end(rangeIn), std::begin(rangeOut));
radix_sort_pair(execPol, std::begin(keysIn), std::begin(valsIn), std::begin(keysOut), std::begin(valsOut), nwork);
```

2. policy(range, execution_body)

```cpp
/// index range [0, 10)
execPol(range(10), execBody);
/// zip range
execPol(zip(ranges...), execBody);
/// enumerate range
execPol(enumerate(ranges...), execBody);
/// 同样支持标准库中的range
execPol(std::vector<float>{...}, execBody);
```



### 执行计算前的设置项

每一个执行策略对象（前述的execPol对象）目前包含以下的设置选项：

* 是否需要同步：execPol.sync(true)

* 是否统计运行时间：execPol.profile(true)

* 错误处理策略（有待实现）：execPol.error(...)

* 特定后端的高级设置（目前主要关于cuda），比如execPol.device(0)等



### 下一步重点研发方向

* 根据某种空间划分策略（hilbert curve等）对数据自动partition以便schedule到多GPU上同时计算运行并同步结果
* 更完备的上下文（context）抽象和实现（参考SYCL的queue）
* 对更大范围内的体系结构（如Arm）提供支持
* 面向分布式系统的并行计算

## ZPC开发维护守则（ZPC develop guideline）

​		ZPC是一个可拓展的框架，允许开发者对各个环节进行拓展，包括但不限于数据结构、存储空间、计算后端、仿真计算等。但由于多后端AOT编译（ahead-of-time compilation）的需要，基于ZPC开发的项目在构建时会依赖宏、c++模板以及c++编译规则等机制。因此ZPC的可拓展性大体建立在静态多态之上，实现中需要针对不同存储空间和执行空间做针对性的模板特化，并依赖tag dispatch等机制在运行时进行模式匹配（pattern matching）调用对应实现。这与传统的c++面向对象编程中基于动态多态的可拓展性有着本质区别。

### API命名规则约定

自由函数或类的static函数：谓词开头、字母全小写、单词间以下划线分隔

非框架核心类：由名词构成、各单词首字母大写、之间无任何符号间隔

框架核心类（如tuple、wrapv等等）：全名词、字母全小写、单词间以下划线分隔

### 存储空间
TBD

### 执行空间
TBD

### 数据结构
TBD

### 自由函数

提供一段模板实现（\*.tpp）以及前置声明（declaration），在各后端的源文件内include该模板实现（\*.tpp）


## ZPC使用样例
TBD

### 开发通用的函数



### 基于空间哈希的邻域查询



### BVH的构建与使用





*Co​n​g​ra​ts​!​ ​Yo​u​'ve ​f​i​ni​sh​e​d ​the crash course! :tada::tada::tada:*
