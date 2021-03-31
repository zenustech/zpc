function(add_cpp_executable binary)
  add_executable(${binary} ${ARGN})
  target_compile_options(${binary} 
    PRIVATE     $<$<COMPILE_LANGUAGE:CXX>: -O3 -fopenmp># -fuse-ld=lld -fvisibility=hidden># -flto=thin -fsanitize=cfi 
  )
  # -fsanitize=address memory thread
  set_target_properties(${binary}
    PROPERTIES  LINKER_LANGUAGE CXX
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(${binary}
    PUBLIC      project_options
                #project_warnings
                project_cxx_dependencies
  )
  message("-- [${binary}]\tcpp executable build config")
endfunction(add_cpp_executable)

function(add_cpp_executable_debug binary)
  add_executable(${binary} ${ARGN})
  target_compile_options(${binary} 
    PRIVATE     $<$<COMPILE_LANGUAGE:CXX>: -O3 -fopenmp># -fuse-ld=lld -fvisibility=hidden -fsanitize=address -fsanitize=memory -fsanitize=thread -fno-sanitize-trap=cfi># -flto=thin -fsanitize=cfi 
  )
  set_target_properties(${binary}
    PROPERTIES  LINKER_LANGUAGE CXX
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(${binary}
    PUBLIC      project_options
                #project_warnings
                project_cxx_dependencies
  )
  message("-- [${binary}]\tcpp debug executable build config")
endfunction(add_cpp_executable_debug)

function(add_cpp_library library)
  add_library(${library} ${ARGN})
  add_library(${project_name}::${library} ALIAS ${library})
  target_compile_options(${library} 
    PRIVATE     $<$<COMPILE_LANGUAGE:CXX>: -O3 -fopenmp># -fuse-ld=lld -fvisibility=hidden># -flto=thin -fsanitize=cfi 
  )
  # -fsanitize=address memory thread
  set_target_properties(${library}
    PROPERTIES  LINKER_LANGUAGE CXX
                POSITION_INDEPENDENT_CODE ON
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(${library}
    PUBLIC      project_options
                #project_warnings
                project_cxx_dependencies
  )
  message("-- [${library}]\tcpp library build config")
endfunction(add_cpp_library)

function(add_shared_cpp_library library)
  add_library(${library} SHARED ${ARGN})
  add_library(${project_name}::${library} ALIAS ${library})
  target_compile_options(${library} 
    PRIVATE     $<$<COMPILE_LANGUAGE:CXX>: -O3 -fopenmp># -fuse-ld=lld -fvisibility=hidden># -flto=thin -fsanitize=cfi 
  )
  # -fsanitize=address memory thread
  set_target_properties(${library}
    PROPERTIES  LINKER_LANGUAGE CXX
                POSITION_INDEPENDENT_CODE ON
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(${library}
    PUBLIC      project_options
                #project_warnings
                project_cxx_dependencies
  )
  message("-- [${library}]\tcpp library build config")
endfunction(add_shared_cpp_library)