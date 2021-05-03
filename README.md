## **Document**
See [git wiki](https://github.com/littlemine/Mn/wiki) page for more details.

## **Formatting**
Please refer to this [repo](https://github.com/barisione/clang-format-hooks) for more info.

Related files are stored within the *Script* folder.

## **Compilation**
This is a cross-platform C++/CUDA cmake project. The minimum version requirement of cmake is 3.18, yet the latest version is generally recommended. Please install cmake through official website or python3-pip, since the cmake version in apt is behind.

The required CUDA version is 11.0+ (for c++17).

Currently, *supported OS* is Ubuntu 20.04, and *tested compilers* includes gcc10.0, clang-11. 

### **Build**

Before building this framework, please first manually configure these external dependencies, i.e. [**openvdb**](https://github.com/AcademySoftwareFoundation/openvdb) and [**umpire**](https://github.com/LLNL/Umpire). 

Be sure to set *ENABLE_CUDA=On* and *ENABLE_NUMA=On* when building **umpire**, see [**advanced umpire build**](https://umpire.readthedocs.io/en/develop/advanced_configuration.html) for more details. Make sure **libnuma** is installed through apt as well.

To build this framework, run the following command in the *root directory*.
```
sudo apt install libeigen3-dev libnuma-dev
sudo python3 -m pip install pybind11
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=g++-10
cmake --build . 
```

Or configure the project using the *CMake Tools* extension in *Visual Studio Code* (recommended).

Certain dependencies are managed by [CPM](https://github.com/TheLartians/CPM.cmake), which downloads the required libraries during the initial cmake build. Please establish a stable proxy connection that can penetrate GFW, otherwise the download process is tremendously slow! 

### **Data**

The *Data* directory, which holds all related assets and materials, is integrated as a submodule. 

Please call
```
git clone git@github.com:littlemine/Mn.git --recursive
or
git clone https://github.com/littlemine/Mn.git --recursive
```
Or if you already cloned the project, call
```
git submodule init
git submodule update
```

## **Code Usage**

> Use the codebase in another cmake c++ project.

Directly include the codebase as a submodule, and follow the examples in the *Projects*.

> Develop upon the codebase.

Create a sub-folder in *Projects* with a cmake file at its root.

Make good use of these predefined CMake functions: add_cpp_executable/ add_cuda_executable, and these internal targets: zensim, ....

> Testing

Same as above, except that do it in *Tests* subdirectory.

## **Credits**
This project draws inspirations from [Taichi](https://github.com/taichi-dev/taichi), [GMPM](https://github.com/kuiwuchn/GPUMPM), kokkos, vsg, raja.

### **Dependencies**
The following libraries are adopted in our project development:

- [openvdb](https://github.com/AcademySoftwareFoundation/openvdb) 
- [umpire](https://github.com/LLNL/Umpire)
- [fmt](https://fmt.dev/latest/index.html)
- spdlog
- cxxopts
- magic_enum
- ...

For particle data IO and generation, we use these two libraries in addition:

- [partio](http://partio.us/)

We import these following libraries as well:

- [function_ref](https://github.com/TartanLlama/function_ref)