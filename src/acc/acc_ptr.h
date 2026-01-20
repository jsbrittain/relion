#ifndef ACC_PTR_H_
#define ACC_PTR_H_

#include "src/acc/settings.h"
#ifdef _CUDA_ENABLED
#include "src/acc/cuda/cuda_settings.h"
#include <cuda_runtime.h>
#include "src/acc/cuda/custom_allocator.cuh"
#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/acc/cuda/shortcuts.cuh"
#include "src/acc/cuda/cuda_pinned_pool.h"
#elif _HIP_ENABLED
#include "src/acc/hip/hip_settings.h"
#include <hip/hip_runtime.h>
#include "src/acc/hip/custom_allocator.h"
#include "src/acc/hip/hip_mem_utils.h"
#include "src/acc/hip/shortcuts.h"
#elif _SYCL_ENABLED
#include <sstream>
#include <string>
#include <cassert>
#include "src/acc/sycl/device_stubs.h"
#include "src/acc/sycl/sycl_settings.h"
#include "src/acc/sycl/sycl_virtual_dev.h"
#else
#include "src/acc/cpu/device_stubs.h"
#include "src/acc/cpu/cpu_settings.h"
#endif

#include <signal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>

#include "src/macros.h"
#include "src/error.h"
#include "src/parallel.h"

#ifndef MEM_ALIGN
	#define MEM_ALIGN 64
#endif

#define ACC_PTR_DEBUG_FATAL( err ) (HandleAccPtrDebugFatal( err, __FILE__, __LINE__ ))
static void HandleAccPtrDebugFatal( const char *err, const char *file, int line )
{
    	fprintf(stderr, "DEBUG ERROR: %s in %s:%d\n", err, file, line );
		fflush(stdout);
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		raise(SIGSEGV);
#else
		CRITICAL(ERRGPUKERN);
#endif
}

#define ACC_PTR_DEBUG_INFO( err ) (HandleAccPtrDebugInformational( err, __FILE__, __LINE__ ))
static void HandleAccPtrDebugInformational( const char *err, const char *file, int line )
{
    	fprintf(stderr, "POSSIBLE ISSUE: %s in %s:%d\n", err, file, line );
		fflush(stdout);
}

enum AccType {accUNSET, accSYCL, accHIP, accCUDA, accCPU};


#ifdef _CUDA_ENABLED
typedef cudaStream_t StreamType;
typedef CudaCustomAllocator AllocatorType;
typedef CudaCustomAllocator::Alloc AllocationType;
#elif _HIP_ENABLED
typedef hipStream_t StreamType;
typedef HipCustomAllocator AllocatorType;
typedef HipCustomAllocator::Alloc AllocationType;
#else
using StreamType = deviceStream_t;
using AllocatorType = double;  //Dummy type
using AllocationType = double;  //Dummy type
#endif

template <typename T>
class AccPtr
{
protected:
	AllocatorType *allocator;
	AllocationType *alloc;
	StreamType stream;
	
	AccType accType;
  bool hostPinned = false;

	size_t size; //Size used when copying data from and to device
  size_t hostCapacity = 0;  // allocated size
	T *hPtr, *dPtr; //Host and device pointers
	bool doFreeDevice; //True if host or device needs to be freed
#ifdef _SYCL_ENABLED
	bool isHostSYCL;    // Check if host pointer is from sycl::malloc_host
#endif

public:
	bool doFreeHost; //TODO make this private
  
  void setHostPinned(bool pinned) {
    hostPinned = pinned;
  }

	/*======================================================
				CONSTRUCTORS WITH ALLOCATORS
	======================================================*/

	AccPtr(AllocatorType *allocator):
		size(0), hPtr(NULL), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL),
#ifdef _CUDA_ENABLED
		stream(cudaStreamPerThread),
		accType(accCUDA)
#elif _HIP_ENABLED
		stream(hipStreamPerThread),
		accType(accHIP)
#else
		stream(cudaStreamPerThread),
		accType(accCPU)
#endif
	{}

	AccPtr(StreamType stream, AllocatorType *allocator):
		size(0), hPtr(NULL), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#elif _HIP_ENABLED
		accType(accHIP)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(size_t size, AllocatorType *allocator):
		size(size), dPtr(NULL), doFreeHost(true),
		doFreeDevice(false), allocator(allocator), alloc(NULL),
#ifdef _CUDA_ENABLED
		stream(cudaStreamPerThread),
		accType(accCUDA)
#elif _HIP_ENABLED
		stream(hipStreamPerThread),
		accType(accHIP)		
#else
		stream(cudaStreamPerThread),
		accType(accCPU)
#endif
	{
		if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
			CRITICAL(RAMERR);
	}

	AccPtr(size_t size, StreamType stream, AllocatorType *allocator):
		size(size), dPtr(NULL), doFreeHost(true),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#elif _HIP_ENABLED
		accType(accHIP)
#else
		accType(accCPU)
#endif
	{
		if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
			CRITICAL(RAMERR);
	}

	AccPtr(T * h_start, size_t size, AllocatorType *allocator):
		size(size), hPtr(h_start), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL), 
#ifdef _CUDA_ENABLED
		stream(cudaStreamPerThread),
		accType(accCUDA)
#elif  _HIP_ENABLED
		stream(hipStreamPerThread),
		accType(accHIP)
#else
		stream(cudaStreamPerThread),
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, size_t size, StreamType stream, AllocatorType *allocator):
		size(size), hPtr(h_start), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL), 
#ifdef _CUDA_ENABLED
		stream(cudaStreamPerThread),
		accType(accCUDA)
#elif _HIP_ENABLED
		stream(hipStreamPerThread),
		accType(accHIP)
#else
		stream(cudaStreamPerThread),
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, T * d_start, size_t size, AllocatorType *allocator):
		size(size), hPtr(h_start), dPtr(d_start), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL),
#ifdef _CUDA_ENABLED
		stream(cudaStreamPerThread),
		accType(accCUDA)
#elif _HIP_ENABLED
		stream(hipStreamPerThread),
		accType(accHIP)
#else
		stream(cudaStreamPerThread),
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, T * d_start, size_t size, StreamType stream, AllocatorType *allocator):
		size(size), hPtr(h_start), dPtr(d_start), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#elif _HIP_ENABLED
		accType(accHIP)
#else
		accType(accCPU)
#endif
	{}

	/*======================================================
	                      CONSTRUCTORS
	======================================================*/

	AccPtr():
		size(0), hPtr(NULL), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL),
#ifdef _CUDA_ENABLED
		stream(cudaStreamPerThread),
		accType(accCUDA)
#elif _HIP_ENABLED
		stream(hipStreamPerThread),
		accType(accHIP)
#else
		stream(cudaStreamPerThread),
		accType(accCPU)
#endif
	{}

	AccPtr(StreamType stream):
		size(0), hPtr(NULL), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#elif _HIP_ENABLED
		accType(accHIP)
#elif _SYCL_ENABLED
		accType(accSYCL)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(size_t size):
		size(size), dPtr(NULL), doFreeHost(true),
		doFreeDevice(false), allocator(NULL), alloc(NULL),
#ifdef _CUDA_ENABLED
		stream(cudaStreamPerThread),
		accType(accCUDA)
#elif _HIP_ENABLED
		stream(hipStreamPerThread),
		accType(accHIP)
#else
		stream(cudaStreamPerThread),
		accType(accCPU)
#endif
	{
		if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
			CRITICAL(RAMERR);
	}

	AccPtr(size_t size, StreamType stream):
		size(size), dPtr(NULL), doFreeHost(true),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#elif _HIP_ENABLED
		accType(accHIP)
#else
		accType(accCPU)
#endif
	{
		if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
			CRITICAL(RAMERR);
	}

	AccPtr(T * h_start, size_t size):
		size(size), hPtr(h_start), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL),
#ifdef _CUDA_ENABLED
		stream(cudaStreamPerThread),
		accType(accCUDA)
#elif _HIP_ENABLED
		stream(hipStreamPerThread),
		accType(accHIP)
#else
		stream(cudaStreamPerThread),
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, size_t size, StreamType stream):
		size(size), hPtr(h_start), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL),
#ifdef _CUDA_ENABLED
		stream(cudaStreamPerThread),
		accType(accCUDA)
#elif _HIP_ENABLED
		stream(hipStreamPerThread),
		accType(accHIP)
#else
		stream(cudaStreamPerThread),
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, T * d_start, size_t size):
		size(size), hPtr(h_start), dPtr(d_start), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL),
#ifdef _CUDA_ENABLED
		stream(cudaStreamPerThread),
		accType(accCUDA)
#elif _HIP_ENABLED
		stream(hipStreamPerThread),
		accType(accHIP)
#else
		stream(cudaStreamPerThread),
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, T * d_start, size_t size, StreamType stream):
		size(size), hPtr(h_start), dPtr(d_start), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#elif _HIP_ENABLED
		accType(accHIP)
#else
		accType(accCPU)
#endif
	{}

	/*======================================================
	       CONSTRUCTORS WITH OTHER POINTERS
	======================================================*/

	AccPtr(const AccPtr &ptr):
		size(ptr.size), hPtr(ptr.hPtr), dPtr(ptr.dPtr), doFreeHost(false),
		doFreeDevice(false), allocator(ptr.allocator), alloc(NULL), stream(ptr.stream),
		accType(ptr.accType)
	{}

	AccPtr(const AccPtr<T> &ptr, size_t start_idx, size_t size):
		size(size), hPtr(&ptr.hPtr[start_idx]), dPtr(&ptr.dPtr[start_idx]), doFreeHost(false),
		doFreeDevice(false), allocator(ptr.allocator), alloc(NULL), stream(ptr.stream),
		accType(ptr.accType)
	{}


	/*======================================================
	                     METHOD BODY
	======================================================*/

	void setAccType(AccType accT)
	{
		accType = accT;
	}

#ifdef _SYCL_ENABLED
	void setStreamAccType(StreamType s, AccType accT = accSYCL)
	{
		stream = s;
		accType = accT;
	}
#endif

	void markReadyEvent()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
	#ifdef DEBUG_CUDA
			if (alloc == NULL)
				ACC_PTR_DEBUG_FATAL("markReadyEvent called on null allocation.\n");
	#endif
			alloc->markReadyEvent(stream);
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
	#ifdef DEBUG_HIP
			if (alloc == NULL)
				ACC_PTR_DEBUG_FATAL("markReadyEvent called on null allocation.\n");
	#endif
			alloc->markReadyEvent(stream);
		}
#endif
	}

	/**
	 * Allocate memory on device
	 */
	void deviceAlloc()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
	#ifdef DEBUG_CUDA
			if(size==0)
				ACC_PTR_DEBUG_FATAL("deviceAlloc called with size == 0");
			if (doFreeDevice)
				ACC_PTR_DEBUG_FATAL("Device double allocation.\n");
	#endif
				doFreeDevice = true;

				alloc = allocator->alloc(size * sizeof(T));
				dPtr = (T*) alloc->getPtr();
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
	#ifdef DEBUG_HIP
			if(size==0)
				ACC_PTR_DEBUG_FATAL("deviceAlloc called with size == 0");
			if (doFreeDevice)
				ACC_PTR_DEBUG_FATAL("Device double allocation.\n");
	#endif
				doFreeDevice = true;

				alloc = allocator->alloc(size * sizeof(T));
				dPtr = (T*) alloc->getPtr();
		}
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			assert(size > 0);
			assert(doFreeDevice == false);

			doFreeDevice = true;
			dPtr = (T*)(stream->syclMalloc(size * sizeof(T), syclMallocType::device));

			if (dPtr == nullptr)
			{
				std::string str = "syclMalloc DEVICE error of size " + std::to_string(size * sizeof(T)) + ".\n";
				ACC_PTR_DEBUG_FATAL(str.c_str());
				CRITICAL(RAMERR);
			}
		}
#endif
	}

	/**
	 * Allocate memory on device with given size
	 */
	void deviceAlloc(size_t newSize)
	{
		size = newSize;
		deviceAlloc();
	}

	/**
	 * Allocate memory on host
	 */
	void hostAlloc()
	{
    if (hostPinned) {
      hostAllocPinned();
      return;
    }
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if(size==0)
			ACC_PTR_DEBUG_FATAL("deviceAlloc called with size == 0");
		if (doFreeHost)
			ACC_PTR_DEBUG_FATAL("Host double allocation.\n");
#endif
		doFreeHost = true;
#ifdef _SYCL_ENABLED
		if(accType == accSYCL)
		{
			hPtr = (T*)(stream->syclMalloc(size * sizeof(T), syclMallocType::host));
			if(hPtr == nullptr)
			{
				std::string str = "devSYCL(" + std::to_string(reinterpret_cast<uintptr_t>(stream)) + " : " + stream->getName() + ")\n";
				str += "syclMalloc HOST error of size " + std::to_string(size * sizeof(T)) + ".\n";

				ACC_PTR_DEBUG_FATAL(str.c_str());
				CRITICAL(RAMERR);
			}
			isHostSYCL = true;
		}
		else
		{
			if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
				CRITICAL(RAMERR);
			isHostSYCL = false;
		}
#else
		// TODO - alternatively, this could be aligned std::vector
		if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
			CRITICAL(RAMERR);
#endif
	}

/*
  // Pinned host allocation (minimal, fallback to posix when CUDA not enabled)
  void hostAllocPinned()
  {
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
      if (size == 0) ACC_PTR_DEBUG_FATAL("hostAllocPinned called with size == 0");
      if (doFreeHost) ACC_PTR_DEBUG_FATAL("Host double allocation (pinned).");
#endif

      void *p = nullptr;

#ifdef _CUDA_ENABLED
      if (cudaMallocHost(&p, sizeof(T) * size) == cudaSuccess) {
          hPtr = (T*)p;
          doFreeHost = true;
          hostPinned = true;
          return;
      }
      throw std::runtime_error("cudaHostAlloc failed");
#endif

      // fallback: regular aligned allocation
      if (posix_memalign(&p, MEM_ALIGN, sizeof(T) * size)) {
          CRITICAL(RAMERR);
      }
      hPtr = (T*)p;
      doFreeHost = true;
      hostPinned = false;
  }
*/

    // Pinned host allocation (minimal, reuse when capacity permits)
  void hostAllocPinned()
  {
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
      if (size == 0) ACC_PTR_DEBUG_FATAL("hostAllocPinned called with size == 0");
      if (doFreeHost) ACC_PTR_DEBUG_FATAL("Host double allocation (pinned).");
#endif

      // If we already have capacity >= requested size, reuse it.
      if (hPtr != nullptr && hostCapacity >= size) {
          // no-op: reuse existing buffer (should not happen when doFreeHost==false,
          // but keep this safe)
          doFreeHost = true;
          hostPinned = hostPinned; // keep existing flag
          return;
      }

      // Try to acquire from the pool (rounding happens inside pool)
      void *p = cuda_pinned_pool_acquire(sizeof(T) * size);
      if (p != nullptr) {
          hPtr = (T*)p;
          hostCapacity = cuda_pinned_pool_get_size(p) / sizeof(T); // get bucket size in elements (if helper available)
          doFreeHost = true;
          hostPinned = true;
          return;
      }
      throw std::runtime_error("cuda_pinned_pool_acquire failed");

#ifdef _CUDA_ENABLED
      // Pool couldn't satisfy. fall back to cudaMallocHost directly
      void* out = nullptr;
      if (cudaMallocHost(&out, sizeof(T) * size) == cudaSuccess) {
          hPtr = (T*)out;
          hostCapacity = size;
          doFreeHost = true;
          hostPinned = true;
          return;
      }
#endif

      // Fallback to posix_memalign if pinned allocation fails
      if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
          CRITICAL(RAMERR);

      hostCapacity = size;
      doFreeHost = true;
      hostPinned = false;
  }

	/**
	 * Allocate memory on host with given size
	 */
	void hostAlloc(size_t newSize)
	{
		size = newSize;
		hostAlloc();
	}

	void hostAllocPinned(size_t newSize)
	{
		size = newSize;
		hostAllocPinned();
	}

	void allAlloc()
	{
		deviceAlloc();
		hostAlloc();
	}

	void allAlloc(size_t newSize)
	{
		size = newSize;
		deviceAlloc();
		hostAlloc();
	}

	void accAlloc()
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
			deviceAlloc();
		else
			hostAlloc();
	}

	void accAlloc(size_t newSize)
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
			deviceAlloc(newSize);
		else
			hostAlloc(newSize);
	}

  /*
	// Allocate storage of a new size for the array
	void resizeHost(size_t newSize)
	{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if (size==0)
			ACC_PTR_DEBUG_INFO("Resizing from size zero (permitted).\n");
#endif
		// TODO - alternatively, this could be aligned std::vector
		T* newArr;
#ifdef _SYCL_ENABLED
		if(accType == accSYCL)
		{
			newArr = (T*)(stream->syclMalloc(newSize * sizeof(T), syclMallocType::host));
			if(newArr == nullptr)
			{
				std::string str = "syclMalloc HOST in resizeHost error of size " + std::to_string(newSize * sizeof(T)) + ".\n";
				ACC_PTR_DEBUG_FATAL(str.c_str());
				CRITICAL(RAMERR);
			}
		}
		else
		{
			if(posix_memalign((void **)&newArr, MEM_ALIGN, sizeof(T) * newSize))
				CRITICAL(RAMERR);
		}
#else
    // if the current buffer is pinned, allocate a new pinned buffer
    if (hostPinned) {
#ifdef _CUDA_ENABLED
        void *p = nullptr;
        if (cudaMallocHost(&p, sizeof(T) * newSize) != cudaSuccess) {
            throw std::runtime_error("cudaHostAlloc failed in resizeHost");
        } else {
            newArr = (T*)p;
        }
#else
        // non-CUDA builds won't have pinned
        if (posix_memalign((void **)&newArr, MEM_ALIGN, sizeof(T) * newSize))
            CRITICAL(RAMERR);
#endif
    } else {
        if (posix_memalign((void **)&newArr, MEM_ALIGN, sizeof(T) * newSize))
            CRITICAL(RAMERR);
    }
#endif

    memset(newArr, 0x0, sizeof(T) * newSize);

    // copy old contents (same as before)...
    if ((size > 0) && (hPtr != NULL)) {
        if (newSize < size)
            memcpy(newArr, hPtr, newSize * sizeof(T));
        else
            memcpy(newArr, hPtr, size * sizeof(T));
        if (newSize > size) {
            size_t theRest = sizeof(T) * (newSize - size);
            memset(((char*)newArr) + (size * sizeof(T)), 0, theRest);
        }
    } else if (hPtr == NULL) {
        memset(newArr, 0x0, sizeof(T) * newSize);
    }

    // free previous buffer using existing logic
    freeHostIfSet();

    // set new pointer and flags
    setSize(newSize);
    setHostPtr(newArr);
    doFreeHost = true;
    // preserve pinnedness if allocation succeeded as pinned
#ifdef _CUDA_ENABLED
    // If hostPinned was true and newArr was created by cudaHostAlloc set hostPinned true,
    // otherwise set to false (posix fallback)
    // Detect via newArr pointer origin is tricky; we can assume if hostPinned previously true
    // we attempted pinned allocation above and set hostPinned accordingly.
    // For simplicity: set hostPinned = hostPinned (i.e. preserve) only if we allocated pinned.
#endif
  }
*/

  void resizeHost(size_t newSize)
  {
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
      if (size==0)
          ACC_PTR_DEBUG_INFO("Resizing from size zero (permitted).\n");
#endif

      // If current host buffer exists and capacity suffices, keep it and adjust size
      if (hPtr != nullptr && hostCapacity >= newSize) {
          // If growing, zero the newly added region
          if (newSize > size) {
              size_t bytes_new_region = (newSize - size) * sizeof(T);
              memset(((char*)hPtr) + size * sizeof(T), 0, bytes_new_region);
          }
          setSize(newSize);
          return;
      }

      // Otherwise allocate a new host buffer (preserve pinnedness preference)
      T* newArr = nullptr;
#ifdef _CUDA_ENABLED
      if (hostPinned) {
          // attempt to allocate a pinned buffer large enough
          void *p = cuda_pinned_pool_acquire(sizeof(T) * newSize);
          if (p != nullptr) {
              newArr = (T*)p;
              hostCapacity = cuda_pinned_pool_get_size(p) / sizeof(T);
              hostPinned = true;
          } else {
            throw std::runtime_error("cuda_pinned_pool_acquire failed in resizeHost");
              // fallback to cudaMallocHost
              void* out = nullptr;
              if (cudaMallocHost(&out, sizeof(T) * newSize) == cudaSuccess) {
                  newArr = (T*)out;
                  hostCapacity = newSize;
                  hostPinned = true;
              } else {
                  // final fallback to posix
                  if (posix_memalign((void **)&newArr, MEM_ALIGN, sizeof(T) * newSize))
                      CRITICAL(RAMERR);
                  hostCapacity = newSize;
                  hostPinned = false;
              }
          }
      } else
#endif
      {
          // non-pinned path
          if (posix_memalign((void **)&newArr, MEM_ALIGN, sizeof(T) * newSize))
              CRITICAL(RAMERR);
          hostCapacity = newSize;
          hostPinned = false;
      }

      // zero initialize full buffer
      memset(newArr, 0x0, sizeof(T) * newSize);

      // copy old contents if any
      if ((size > 0) && (hPtr != NULL)) {
          if (newSize < size)
              memcpy(newArr, hPtr, newSize * sizeof(T));
          else
              memcpy(newArr, hPtr, size * sizeof(T));
      }

      // free previous buffer using existing logic
      freeHostIfSet();

      // set new pointer and flags
      setSize(newSize);
      setHostPtr(newArr);
      doFreeHost = true;
  }

	// Resize retaining as much of the original contents as possible
	void resizeHostCopy(size_t newSize)
	{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
//		if (size==0)
//			ACC_PTR_DEBUG_INFO("Resizing from size zero (permitted).\n");
#endif
		// TODO - alternatively, this could be aligned std::vector
		T* newArr;
#ifdef _SYCL_ENABLED
		if(accType == accSYCL)
		{
			newArr = (T*)(stream->syclMalloc(newSize * sizeof(T), syclMallocType::host));
			if(newArr == nullptr)
			{
				std::string str = "syclMalloc HOST in resizeHostCopy error of size " + std::to_string(newSize * sizeof(T)) + ".\n";
				ACC_PTR_DEBUG_FATAL(str.c_str());
				CRITICAL(RAMERR);
			}
		}
		else
		{
			if(posix_memalign((void **)&newArr, MEM_ALIGN, sizeof(T) * newSize))
				CRITICAL(RAMERR);
		}
#else
		if(posix_memalign((void **)&newArr, MEM_ALIGN, sizeof(T) * newSize))
			CRITICAL(RAMERR);
#endif
		
		// Copy in what we can from the original matrix
		if ((size > 0) && (hPtr != NULL))
		{
			if (newSize < size)
				memcpy( newArr, hPtr, newSize * sizeof(T) );
			else
				memcpy( newArr, hPtr, size * sizeof(T) );  
			
			// Initialize remaining memory if any
			if (newSize > size)
			{
				size_t theRest = sizeof(T) * (newSize - size);
				memset( newArr, 0x0, theRest);
			}
		}
		
		// There was nothing from before to copy - clear new memory
		if (hPtr == NULL)
		{
			memset( newArr, 0x0, sizeof(T) * newSize);
		}

#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if (dPtr!=NULL)
			ACC_PTR_DEBUG_FATAL("resizeHostCopy: Resizing host with present device allocation.\n");
		if (newSize==0)
			ACC_PTR_DEBUG_INFO("resizeHostCopy: Array resized to size zero (permitted with fear).  Something may break downstream\n");
#endif
		freeHostIfSet();
	    setSize(newSize);
	    setHostPtr(newArr);
#ifdef _SYCL_ENABLED
		if(accType == accSYCL)
			isHostSYCL = true;
		else
			isHostSYCL = false;
#endif
	    doFreeHost=true;
	}
	
	/**
	 * Initiate device memory with provided value
	 */
	void deviceInit(int value)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
	#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("Memset requested before allocation in deviceInit().\n");
	#endif
			cudaMemInit<T>( dPtr, value, size, stream);
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
	#ifdef DEBUG_HIP
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("Memset requested before allocation in deviceInit().\n");
	#endif
			hipMemInit<T>( dPtr, value, size, stream);
		}
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			assert(dPtr != NULL);
			assert(value == 0);

			stream->syclMemset(dPtr, value, size * sizeof(T));
		}
#endif
	}

	/**
	 * Initiate host memory with provided value
	 */
	void hostInit(int value)
	{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("Memset requested before allocation in hostInit().\n");
#endif
		memset(hPtr, value, size * sizeof(T));
	}

	/**
	 * Initiate memory with provided value
	 */
	void accInit(int value)
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
			deviceInit(value);
		else
			hostInit(value);
	}

	/**
	 * Initiate all used memory with provided value
	 */
	void allInit(int value)
	{
		hostInit(value);
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
			deviceInit(value);
	}

	/**
	 * Copy a number (size) of bytes to device stored in the host pointer
	 */
	void cpToDevice()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
	#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cpToDevice() called before allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cpToDevice().\n");
	#endif
			CudaShortcuts::cpyHostToDevice<T>(hPtr, dPtr, size, stream);
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
	#ifdef DEBUG_HIP
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cpToDevice() called before allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cpToDevice().\n");
	#endif
			HipShortcuts::cpyHostToDevice<T>(hPtr, dPtr, size, stream);
		}
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			assert(dPtr != NULL);
			assert(hPtr != NULL);

			stream->syclMemcpy(dPtr, hPtr, size * sizeof(T));
		}
#endif
	}

	/**
	 * Copy a number (size) of bytes to device stored in the provided host pointer
	 */
	void cpToDevice(T * hostPtr)
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
		{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
			if (hostPtr == NULL)
				ACC_PTR_DEBUG_FATAL("Null-pointer given in cpToDevice(hostPtr).\n");
#endif
			hPtr = hostPtr;
			cpToDevice();
		}
	}

	/**
	 * alloc and copy
	 */
	void putOnDevice()
	{
		deviceAlloc();
		cpToDevice();
	}

	/**
	 * alloc size and copy
	 */
	void putOnDevice(size_t newSize)
	{
		size=newSize;
		deviceAlloc();
		cpToDevice();
	}


	/**
	 * Copy a number (size) of bytes from device to the host pointer
	 */
	void cpToHost()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
	#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host() called before device allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host().\n");
	#endif
			cudaCpyDeviceToHost<T>(dPtr, hPtr, size, stream);
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
	#ifdef DEBUG_HIP
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host() called before device allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host().\n");
	#endif
			hipCpyDeviceToHost<T>(dPtr, hPtr, size, stream);
		}
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			assert(dPtr != NULL);
			assert(hPtr != NULL);

			stream->syclMemcpy(hPtr, dPtr, size * sizeof(T));
		}
#endif
	}

	/**
	 * Copy a number (thisSize) of bytes from device to the host pointer
	 */
	void cpToHost(size_t thisSize)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
	#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host(thisSize) called before device allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host(thisSize).\n");
	#endif
			cudaCpyDeviceToHost<T>(dPtr, hPtr, thisSize, stream);
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
	#ifdef DEBUG_HIP
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host(thisSize) called before device allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host(thisSize).\n");
	#endif
			hipCpyDeviceToHost<T>(dPtr, hPtr, thisSize, stream);
		}
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			assert(dPtr != NULL);
			assert(hPtr != NULL);

			stream->syclMemcpy(hPtr, dPtr, thisSize * sizeof(T));
		}
#endif
	}

	/**
	 * Copy a number (thisSize) of bytes from device to a specific host pointer
	 */
	void cpToHost(T* hstPtr, size_t thisSize)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
	#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host(hstPtr, thisSize) called before device allocation.\n");
			if (hstPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host(hstPtr, thisSize).\n");
	#endif
			cudaCpyDeviceToHost<T>(dPtr, hstPtr, thisSize, stream);
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
	#ifdef DEBUG_HIP
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host(hstPtr, thisSize) called before device allocation.\n");
			if (hstPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host(hstPtr, thisSize).\n");
	#endif
			hipCpyDeviceToHost<T>(dPtr, hstPtr, thisSize, stream);
		}
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			assert(dPtr != NULL);
			assert(hstPtr != NULL);
			
			stream->syclMemcpy(hstPtr, dPtr, thisSize * sizeof(T));
		}
#endif
	}

	/**
	 * Copy a number (size) of bytes from device to the host pointer
	 */
	void cpToHostOnStream(StreamType s)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
	#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host_on_stream(s) called before device allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host_on_stream(s).\n");
	#endif
			cudaCpyDeviceToHost<T>(dPtr, hPtr, size, s);
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
	#ifdef DEBUG_HIP
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host_on_stream(s) called before device allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host_on_stream(s).\n");
	#endif
			hipCpyDeviceToHost<T>(dPtr, hPtr, size, s);
		}
#elif _SYCL_ENABLED
		if (accType == accSYCL)
			ACC_PTR_DEBUG_FATAL("cpToHostOnStream(StreamType s) does not work on SYCL.\n");
#endif
	}

	/**
	 * Copy a number (size) of bytes from device pointer to the provided new device pointer
	 */
	void cpOnDevice(T * dstDevPtr)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
	#ifdef DEBUG_CUDA
			if (dstDevPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL-pointer given in cpOnDevice(dstDevPtr).\n");
	#endif
			CudaShortcuts::cpyDeviceToDevice(dPtr, dstDevPtr, size, stream);
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
	#ifdef DEBUG_HIP
			if (dstDevPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL-pointer given in cpOnDevice(dstDevPtr).\n");
	#endif
			HipShortcuts::cpyDeviceToDevice(dPtr, dstDevPtr, size, stream);
		}
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			assert(dPtr != NULL);
			assert(dstDevPtr != NULL);

			stream->syclMemcpy(dstDevPtr, dPtr, size * sizeof(T));
		}
#endif
	}

	/**
	 * Copy a number (size) of bytes from host pointer to the provided new host pointer
	 */
	void cpOnHost(T * dstDevPtr)
	{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if (dstDevPtr == NULL)
			ACC_PTR_DEBUG_FATAL("NULL-pointer given in cp_on_host(dstDevPtr).\n");
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("NULL input pointer given in cp_on_host(hPtr).\n");
#endif
		memcpy ( dstDevPtr, hPtr, size * sizeof(T));
	}

	void cpOnAcc(T * dstDevPtr)
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
			cpOnDevice(dstDevPtr);
		else
			cpOnHost(dstDevPtr);
	}

	void cpOnAcc(AccPtr<T> &devPtr)
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
			cpOnDevice(devPtr.dPtr);
		else
			cpOnHost(devPtr.hPtr);
	}

	/**
	 * Host data quick access
	 */
	const T& operator[](size_t idx) const
	{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("const operator[] called with NULL host pointer.\n");
#endif
		return hPtr[idx];
	};

	/**
	 * Host data quick access
	 */
	T& operator[](size_t idx)
	{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("operator[] called with NULL host pointer.\n");
#endif
		return hPtr[idx];
	};
	
	/**
	 * Device data quick access
	 */
	T& operator()(size_t idx) 
	{ 
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("operator(idx) called with NULL acc pointer.\n");
#endif
		return dPtr[idx]; 
	};


	/**
	 * Device data quick access
	 */
	const T& operator()(size_t idx) const 
	{ 
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("operator(idx) called with NULL acc pointer.\n");
#endif
		return dPtr[idx]; 
	};
	
	/**
	 * Raw data pointer quick access
	 */
	T* operator()()
	{
		// TODO - this could cause considerable confusion given the above operators.  But it
		// also simplifies code that uses it.   What to do...
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
		{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("operator() called with NULL device pointer.\n");
#endif
			return dPtr;
		}
		else
		{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("operator() called with NULL host pointer.\n");
#endif
			return hPtr;
		}
	};

	T* operator~() 
	{
		// TODO - this could cause considerable confusion given the above operators.  But it
		// also simplifies code that uses it.   What to do...
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
		{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
			if ( dPtr == 0)
				ACC_PTR_DEBUG_FATAL("DEBUG_WARNING: \"kernel cast\" on null device pointer.\n");
#endif
			return dPtr;
		}
		else
		{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if ( hPtr == 0)
			ACC_PTR_DEBUG_FATAL("DEBUG_WARNING: \"kernel cast\" on null host pointer.\n");
#endif
		return hPtr;
		}
	}
	
	void streamSync()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
#elif _HIP_ENABLED
		if (accType == accHIP)
			DEBUG_HANDLE_ERROR(hipStreamSynchronize(stream));
#elif _SYCL_ENABLED
		if (accType == accSYCL)
			stream->waitAll();
#endif
	}

	T getAccValueAt(size_t idx)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
			T value;
			cudaCpyDeviceToHost<T>(&dPtr[idx], &value, 1, stream);
			streamSync();
			return value;
		}
		else
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
			T value;
			hipCpyDeviceToHost<T>(&dPtr[idx], &value, 1, stream);
			streamSync();
			return value;
		}
		else
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			assert(dPtr != NULL);
			assert(idx < size);
			T value;
			stream->syclMemcpy(&value, &dPtr[idx], sizeof(T));
			streamSync();
			return value;
		}
		else
#endif
			return hPtr[idx];
	}

	T getDeviceAt(size_t idx)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
			T value;
			cudaCpyDeviceToHost<T>(&dPtr[idx], &value, 1, stream);
			streamSync();
			return value;
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
			T value;
			hipCpyDeviceToHost<T>(&dPtr[idx], &value, 1, stream);
			streamSync();
			return value;
		}
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			assert(dPtr != NULL);
			assert(idx < size);
			T value;
			stream->syclMemcpy(&value, &dPtr[idx], sizeof(T));
			streamSync();
			return value;
		}
#else
		return NULL;
#endif
	}

	void setAccValueAt(T value, size_t idx)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
			CudaShortcuts::cpyHostToDevice<T>(&value, &dPtr[idx], sizeof(T), stream);
			streamSync();
		}
		else
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
			HipShortcuts::cpyHostToDevice<T>(&value, &dPtr[idx], sizeof(T), stream);
			streamSync();
		}
		else
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			assert(dPtr != NULL);
			assert(idx < size);
			stream->syclMemcpy(&dPtr[idx], &value, sizeof(T));
			streamSync();
		}
		else
#endif
			hPtr[idx] = value;
	}

	void dumpDeviceToFile(std::string fileName)
	{

#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
			T *tmp = new T[size];
			cudaCpyDeviceToHost<T>(dPtr, tmp, size, stream);

			std::ofstream f;
			f.open(fileName.c_str());
			streamSync();
			for (unsigned i = 0; i < size; i ++)
				f << tmp[i] << std::endl;
			f.close();
			delete [] tmp;
		}
		else
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
			T *tmp = new T[size];
			hipCpyDeviceToHost<T>(dPtr, tmp, size, stream);

			std::ofstream f;
			f.open(fileName.c_str());
			streamSync();
			for (unsigned i = 0; i < size; i ++)
				f << tmp[i] << std::endl;
			f.close();
			delete [] tmp;
		}
		else
#elif _SYCL_ENABLED
		if (accType == accSYCL)
		{
			T *tmp = new T[size];
			stream->syclMemcpy(tmp, dPtr, size * sizeof(T));

			std::ofstream f;
			f.open(fileName.c_str());
			streamSync();
			for (unsigned i = 0; i < size; i ++)
				f << tmp[i] << std::endl;
			f.close();
			delete [] tmp;
		}
		else
#endif
		{
			std::ofstream f;
			f.open(fileName.c_str());
			f << "Pointer has no device support." << std::endl;
			f.close();
		}
	}

	void dumpHostToFile(std::string fileName)
	{
		std::ofstream f;
		f.open(fileName.c_str());
		for (unsigned i = 0; i < size; i ++)
			f << hPtr[i] << std::endl;
		f.close();
	}

	void dumpAccToFile(std::string fileName)
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
			dumpDeviceToFile(fileName);
		else
			dumpHostToFile(fileName);
	}

	/**
	 * Delete device data
	 */
	void freeDevice()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
	#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("Free device memory was called on NULL pointer in free_device().\n");
	#endif
			doFreeDevice = false;

			if (alloc->getReadyEvent() == 0)
				alloc->markReadyEvent(stream);
			alloc->doFreeWhenReady();
			alloc = NULL;

//			DEBUG_HANDLE_ERROR(cudaFree(dPtr));

			dPtr = NULL;
		}
#elif _HIP_ENABLED
		if (accType == accHIP)
		{
	#ifdef DEBUG_HIP
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("Free device memory was called on NULL pointer in free_device().\n");
	#endif
			doFreeDevice = false;

			if (alloc->getReadyEvent() == 0)
				alloc->markReadyEvent(stream);
			alloc->doFreeWhenReady();
			alloc = NULL;

//			DEBUG_HANDLE_ERROR(hipFree(dPtr));

			dPtr = NULL;
		}
#elif _SYCL_ENABLED
//		if (accType == accSYCL)
		{
			assert(dPtr != NULL);

			stream->waitAll();
			stream->syclFree(dPtr);
			doFreeDevice = false;
			dPtr = NULL;
		}
#endif
	}

	/**
	 * Delete host data
	 */
	void freeHost()
  {
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
      if (hPtr == NULL)
          ACC_PTR_DEBUG_FATAL("free_host() called on NULL pointer.\n");
#endif
      doFreeHost = false;
      if (NULL != hPtr)
#ifdef _SYCL_ENABLED
      {
          if(isHostSYCL)
              stream->syclFree(hPtr);
          else
              free(hPtr);
      }
#else
      {
#ifdef _CUDA_ENABLED
          if (hostPinned) {
              //cudaFreeHost(hPtr);
              cuda_pinned_pool_release(hPtr);
          } else {
              free(hPtr);
          }
#else
          free(hPtr);
#endif
      }
#endif
      hPtr = NULL;
      hostPinned = false;
  }

	void freeHostIfSet()
	{
		if (doFreeHost)
			freeHost();
	}

	void freeDeviceIfSet()
	{
		if (doFreeDevice)
			freeDevice();
	}

	/**
	 * Delete both device and host data
	 */
	void freeBoth()
	{
		freeDevice();
		freeHost();
	}

	void freeIfSet()
	{
		freeHostIfSet();
		freeDeviceIfSet();
	}

	~AccPtr()
	{
		freeIfSet();
	}


	/*======================================================
	                   GETTERS AND SETTERS
	======================================================*/


	bool willFreeHost()
	{
		return doFreeHost;
	}

	bool willFreeDevice()
	{
		return doFreeDevice;
	}

	void setStream(StreamType s)
	{
		stream = s;
	}

	StreamType getStream()
	{
		return stream;
	}

	void setSize(size_t s)
	{
		size = s;
	}
	
	size_t getSize()
	{
		return size;
	}

	T *getDevicePtr()
	{
		return dPtr;
	}

	T *getHostPtr()
	{
		return hPtr;
	}

	T *getAccPtr()
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
			return dPtr;
		else
			return hPtr;
	}

	void setAllocator(AllocatorType *a)
	{
		freeDeviceIfSet();
		allocator = a;
	};

	AllocatorType *getAllocator()
	{
		return allocator;
	}

	void setDevicePtr(T *ptr)
	{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
			if (doFreeDevice)
				ACC_PTR_DEBUG_FATAL("Device pointer set without freeing the old one.\n");
#endif
		dPtr = ptr;
	}

	void setDevicePtr(const AccPtr<T> &ptr)
	{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if (ptr.dPtr == NULL)
			ACC_PTR_DEBUG_FATAL("Device pointer is not set.\n");
#endif
		setDevicePtr(ptr.dPtr);
	}

	void setHostPtr(T *ptr)
	{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if (doFreeHost)
			ACC_PTR_DEBUG_FATAL("Host pointer set without freeing the old one.\n");
#endif
		hPtr = ptr;
	}

  void setHostPtr(T *ptr, size_t capacity_in_elements, bool pinned=false)
  {
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
      if (doFreeHost)
          ACC_PTR_DEBUG_FATAL("Host pointer set without freeing the old one.\n");
#endif
      hPtr = ptr;
      hostCapacity = capacity_in_elements;
      hostPinned = pinned;
  }

	void setHostPtr(const AccPtr<T> &ptr)
	{
#if defined(DEBUG_CUDA) || defined(DEBUG_HIP)
		if (ptr.hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("Host pointer is not set.\n");
#endif
		setHostPtr(ptr.hPtr);
	}

	void setAccPtr(const AccPtr<T> &ptr)
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
			setDevicePtr(ptr.hPtr);
		else
			setHostPtr(ptr.hPtr);
	}

	void setAccPtr(T *ptr)
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
			setDevicePtr(ptr);
		else
			setHostPtr(ptr);
	}

	AccType getAccType()
	{
		return accType;
	}

	template <typename Tn>
	AccPtr<Tn> make()
	{
		AccPtr<Tn> ptr(stream, allocator);
		ptr.setAccType(accType);

		return ptr;
	}

	template <typename Tn>
	AccPtr<Tn> make(size_t s)
	{
		AccPtr<Tn> ptr(stream, allocator);
		ptr.setAccType(accType);
		ptr.setSize(s);

		return ptr;
	}
};

typedef unsigned char AccPtrBundleByte;

class AccPtrBundle: public AccPtr<AccPtrBundleByte>
{
private:
	size_t current_packed_pos;

public:
	AccPtrBundle(StreamType stream, AllocatorType *allocator):
		AccPtr<AccPtrBundleByte>(stream, allocator),
		current_packed_pos(0)
	{}

	AccPtrBundle(size_t size, StreamType stream, AllocatorType *allocator):
		AccPtr<AccPtrBundleByte>(stream, allocator),
		current_packed_pos(0)
	{
		setSize(size);
	}

#ifdef _SYCL_ENABLED
	AccPtrBundle(StreamType dev):
		AccPtr<AccPtrBundleByte>(dev), current_packed_pos(0)
	{}
#endif

	template <typename T>
	void pack(AccPtr<T> &ptr)
	{
		if (accType == accCUDA || accType == accHIP || accType == accSYCL)
		{
	#if defined DEBUG_CUDA || defined DEBUG_HIP
			if (current_packed_pos + ptr.getSize() > size)
				ACC_PTR_DEBUG_FATAL("Packing exceeds bundle total size.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("Pack called on null host pointer.\n");
	#endif
			if (ptr.getHostPtr() != NULL)
				memcpy ( &hPtr[current_packed_pos], ptr.getHostPtr(), ptr.getSize() * sizeof(T));
			ptr.freeIfSet();
			ptr.setHostPtr((T*) &hPtr[current_packed_pos]);
			ptr.setDevicePtr((T*) &dPtr[current_packed_pos]);

			current_packed_pos += ptr.getSize() * sizeof(T);
		}
		else
		{
			if (ptr.getHostPtr() == NULL)
				ptr.hostAlloc();
		}
	}
	
	//Overwrite allocation methods and block for no device
	
	void allAlloc()
	{
#if defined _CUDA_ENABLED || defined _HIP_ENABLED || defined _SYCL_ENABLED
		AccPtr<AccPtrBundleByte>::allAlloc();
#endif
	}
	
	void allAlloc(size_t size)
	{
#if defined _CUDA_ENABLED || defined _HIP_ENABLED || defined _SYCL_ENABLED
		AccPtr<AccPtrBundleByte>::allAlloc(size);
#endif
	}
	
	void hostAlloc()
	{
#if defined _CUDA_ENABLED || defined _HIP_ENABLED || defined _SYCL_ENABLED
AccPtr<AccPtrBundleByte>::hostAlloc();
#endif
	}
	
	void hostAlloc(size_t size)
	{
#if defined _CUDA_ENABLED || defined _HIP_ENABLED || defined _SYCL_ENABLED
AccPtr<AccPtrBundleByte>::hostAlloc(size);
#endif
	}
	
};

class AccPtrFactory
{
private:
	AllocatorType *allocator;
	StreamType stream;

	AccType accType;

public:
	AccPtrFactory():
		allocator(NULL), stream(0), accType(accUNSET)
	{}

	AccPtrFactory(AccType accT):
		allocator(NULL), stream(0), accType(accT)
	{}

#ifdef _SYCL_ENABLED
	AccPtrFactory(StreamType s):
		allocator(NULL), stream(s), accType(accSYCL)
	{}
#else
	AccPtrFactory(AllocatorType *alloc):
#ifdef _CUDA_ENABLED
		allocator(alloc), stream(0), accType(accCUDA)
#elif _HIP_ENABLED
		allocator(alloc), stream(0), accType(accHIP)
#else
		allocator(alloc), stream(0), accType(accCPU)
#endif
	{}

	AccPtrFactory(AllocatorType *alloc, StreamType s):
#ifdef _CUDA_ENABLED
		allocator(alloc), stream(s), accType(accCUDA)
#elif _HIP_ENABLED
		allocator(alloc), stream(s), accType(accHIP)
#else
		allocator(alloc), stream(0), accType(accCPU)
#endif
	{}
#endif

	template <typename T>
	AccPtr<T> make()
	{
		if (accType == accSYCL)
		{
			AccPtr<T> ptr(stream);
			ptr.setAccType(accType);

			return ptr;
		}
		else
		{
			AccPtr<T> ptr(stream, allocator);
			ptr.setAccType(accType);

			return ptr;
		}
	}

	template <typename T>
	AccPtr<T> make(size_t size)
	{
		if (accType == accSYCL)
		{
			AccPtr<T> ptr(stream);
			ptr.setAccType(accType);
			ptr.setSize(size);

			return ptr;
        }
        else
        {
			AccPtr<T> ptr(stream, allocator);
			ptr.setAccType(accType);
			ptr.setSize(size);

			return ptr;
		}
	}


	template <typename T>
	AccPtr<T> make(size_t size, StreamType s)
	{
		AccPtr<T> ptr(s, allocator);
		ptr.setAccType(accType);
		ptr.setSize(size);

		return ptr;
	}

  template <typename T>
  AccPtr<T> make_pinned()
  {
      AccPtr<T> ptr(stream, allocator);
      ptr.setAccType(accType);
      ptr.setHostPinned(true);
      return ptr;
  }

  template <typename T>
  AccPtr<T> make_pinned(size_t size)
  {
      AccPtr<T> ptr(stream, allocator);
      ptr.setAccType(accType);
      ptr.setSize(size);
      ptr.setHostPinned(true);
      return ptr;
  }

  template <typename T>
  AccPtr<T> make_pinned(size_t size, StreamType s)
  {
      AccPtr<T> ptr(s, allocator);
      ptr.setAccType(accType);
      ptr.setSize(size);
      ptr.setHostPinned(true);
      return ptr;
  }

	AccPtrBundle makeBundle()
	{
		if (accType == accSYCL)
		{
			AccPtrBundle bundle(stream, 0);
			bundle.setAccType(accType);

			return bundle;
		}
		else
		{
			AccPtrBundle bundle(stream, allocator);
			bundle.setAccType(accType);

			return bundle;
		}
	}

	AccPtrBundle makeBundle(size_t size)
	{
		AccPtrBundle bundle(size, stream, allocator);
		bundle.setAccType(accType);

		return bundle;
	}

#ifdef _SYCL_ENABLED
	template <typename T>
	AccPtr<T> make(StreamType dev)
	{
		AccPtr<T> ptr(dev);
		ptr.setAccType(accType);

		return ptr;
	}

	AccPtrBundle makeBundle(StreamType dev)
	{
		AccPtrBundle bundle(dev);
		bundle.setAccType(accType);

		return bundle;
	}
#endif
};

#endif
