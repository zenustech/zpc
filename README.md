# Zenus Simulation
*Zenus Simulation* is the codebase **zensim** maintained by **Zenus Tech**, which delivers great parallel computing efficiency for physics-based simulations within a shared-memory heterogeneous architecture through a unified programming interface on multiple compute backends.

This repo is going through rapid changes, and we do not promise ABI compatibility from commit to commit for the moment.

## **Document**
See [git wiki](https://github.com/zensim-dev/zpc/wiki) page for more build details.
See [Specification](Specification.md) for more usage info.

## **Compilation**
This is a cross-platform C++/CUDA cmake project. The minimum version requirement of cmake is 3.18, yet the latest version is generally recommended. Please install cmake through official website or python3-pip, since the cmake version in apt repo is behind.

When CUDA is enabled, the required CUDA version is 11.4+ (for c++17 and latest cuda utilities).

Currently, *supported OS* are Ubuntu 20.04+ and Windows 10, and *tested compilers* includes gcc10.0+, clang-11+, vs2019+. 

### **Build**

Before building this framework, please first manually configure these external dependencies, i.e. [**openvdb**](https://github.com/AcademySoftwareFoundation/openvdb) if ZS_ENABLE_OPENVDB is set to TRUE. Then pull all dependencies by

```
git submodule init
git submodule update
```

If CUDA (>=11.4) is installed and required, be sure to set *ZS_ENABLE_CUDA=On* first.

Configure the project using the *CMake Tools* extension in *Visual Studio Code* (recommended), or follow the build instructions in [git wiki](https://github.com/zensim-dev/zpc/wiki). 

In addition, make sure to install *zlib* for building *partio* when building on linux.
```
sudo apt install zlib1g
```

## **Integration**

Directly include the codebase as a submodule. Or install this repo then use *find_package(zensim)*.
If the installed package is no longer needed, build the *uninstall* target as with the *install* target.

## **Credits**
This framework draws inspirations from [Taichi](https://github.com/taichi-dev/taichi), [kokkos](https://github.com/kokkos/kokkos), [raja](https://github.com/LLNL/RAJA), [MGMPM](https://github.com/penn-graphics-research/claymore), [GPU LBVH](https://github.com/littlemine/BVH-based-Collision-Detection-Scheme).

### **Dependencies**
The following libraries are adopted and made fully localized in our project development:
- [fmt](https://fmt.dev/latest/index.html)
- [loguru](https://github.com/emilk/loguru). (pro: "chrono" is not exposed in its header)
- [magic_enum](https://github.com/Neargye/magic_enum)
- [catch2](https://github.com/catchorg/Catch2)

For spatial data IO and generation, we use these libraries in addition:

- [partio](http://partio.us/)
- [openvdb](https://github.com/AcademySoftwareFoundation/openvdb) 

We import these (less-frequently used) libraries as well:

- [function_ref](https://github.com/TartanLlama/function_ref)
- [rapidjson](https://github.com/Tencent/rapidjson)
- [cxxopts](https://github.com/jarro2783/cxxopts)
- [jitify](https://github.com/NVIDIA/jitify)
