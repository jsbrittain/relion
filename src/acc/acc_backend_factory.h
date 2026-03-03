#ifndef ACC_BACKEND_FACTORY_H_
#define ACC_BACKEND_FACTORY_H_

#include <memory>
#include "src/acc/acc_backend.h"

#if defined _CUDA_ENABLED || defined _HIP_ENABLED
#include "src/acc/cuda/cuda_backend.h"
#endif
#ifdef _SYCL_ENABLED
#include "src/acc/sycl/sycl_backend.h"
#endif
#include "src/acc/cpu/cpu_backend.h"

inline std::unique_ptr<AccBackend> makeAccBackend(bool do_gpu, bool do_sycl, bool do_cpu)
{
#if defined _CUDA_ENABLED || defined _HIP_ENABLED
    if (do_gpu)  return std::unique_ptr<AccBackend>(new CudaBackend());
#endif
#ifdef _SYCL_ENABLED
    if (do_sycl) return std::unique_ptr<AccBackend>(new SyclBackend());
#endif
#ifdef ALTCPU
    if (do_cpu)  return std::unique_ptr<AccBackend>(new AltCpuBackend());
#endif
    return std::unique_ptr<AccBackend>(new PlainCpuBackend());
}

#endif /* ACC_BACKEND_FACTORY_H_ */
