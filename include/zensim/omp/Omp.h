#pragma once

#if !defined(ZS_ENABLE_OPENMP) || defined(ZS_ENABLE_OPENMP) && ZS_ENABLE_OPENMP == 0
#  error "ZS_ENABLE_OPENMP was not enabled, but Omp.h was included anyway."
#endif

#if !defined(ZPC_JIT_MODE)
#  include <omp.h>
#else

#  if defined(_WIN32)
#    define ZPC_OMP_C_CONVENTION __cdecl
#    ifndef __KMP_IMP
#      define __KMP_IMP __declspec(dllimport)
#    endif
#  else
#    define ZPC_OMP_C_CONVENTION
#    ifndef __KMP_IMP
#      define __KMP_IMP
#    endif
#  endif

#  ifdef __cplusplus
extern "C" {
#  endif

/* set API functions */
void ZPC_OMP_C_CONVENTION omp_set_num_threads(int);
void ZPC_OMP_C_CONVENTION omp_set_dynamic(int);
void ZPC_OMP_C_CONVENTION omp_set_nested(int);
void ZPC_OMP_C_CONVENTION omp_set_max_active_levels(int);

/* query API functions */
int ZPC_OMP_C_CONVENTION omp_get_num_threads(void);
int ZPC_OMP_C_CONVENTION omp_get_dynamic(void);
int ZPC_OMP_C_CONVENTION omp_get_nested(void);
int ZPC_OMP_C_CONVENTION omp_get_max_threads(void);
int ZPC_OMP_C_CONVENTION omp_get_thread_num(void);
int ZPC_OMP_C_CONVENTION omp_get_num_procs(void);
int ZPC_OMP_C_CONVENTION omp_in_parallel(void);
int ZPC_OMP_C_CONVENTION omp_in_final(void);
int ZPC_OMP_C_CONVENTION omp_get_active_level(void);
int ZPC_OMP_C_CONVENTION omp_get_level(void);
int ZPC_OMP_C_CONVENTION omp_get_ancestor_thread_num(int);
int ZPC_OMP_C_CONVENTION omp_get_team_size(int);
int ZPC_OMP_C_CONVENTION omp_get_thread_limit(void);
int ZPC_OMP_C_CONVENTION omp_get_max_active_levels(void);
int ZPC_OMP_C_CONVENTION omp_get_max_task_priority(void);

#  ifdef __cplusplus
}
#  endif
#endif  // end ZPC_JIT_MODE