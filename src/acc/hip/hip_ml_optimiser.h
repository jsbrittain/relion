/* Portions of this code are under:
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#ifndef HIP_ML_OPTIMISER_H_
#define HIP_ML_OPTIMISER_H_

#include "src/mpi.h"
#include "src/ml_optimiser.h"
#include "src/acc/hip/hip_mem_utils.h"
#include "src/acc/acc_projector_plan.h"
#include "src/acc/acc_projector.h"
#include "src/acc/acc_backprojector.h"
#include "src/acc/hip/hip_fft.h"
#include "src/acc/hip/hip_benchmark_utils.h"
#include "src/acc/acc_bundle.h"
#include "src/acc/acc_optimiser.h"
#include <stack>
//#include <hipfft.h>

#include "src/acc/acc_ml_optimiser.h"
#include "src/acc/acc_ptr.h"

class MlDeviceBundle : public AccBundle
{
public:

	//Used for precalculations of projection setup
	HipCustomAllocator *allocator;

	MlOptimiser *baseMLO;

	int device_id;

	int rank_shared_count;

	bool haveWarnedRefinementMem;

	MlDeviceBundle(MlOptimiser *baseMLOptimiser):
			baseMLO(baseMLOptimiser),
			rank_shared_count(1),
			device_id(-1),
			haveWarnedRefinementMem(false),
			allocator(NULL)
	{
		generateProjectionPlanOnTheFly = false;
	};

	void setDevice(int did)
	{
		device_id = did;
	}

	size_t checkFixedSizedObjects(int shares);
	void setupFixedSizedObjects();
	void setupTunableSizedObjects(size_t allocationSize);

	void syncAllBackprojects() override
	{
		DEBUG_HANDLE_ERROR(hipDeviceSynchronize());
	}

	void extractAndAccumulate(MlWsumModel &wsum) override
	{
		for (int j = 0; j < (int)backprojectors.size(); j++)
		{
			unsigned long s = wsum.BPref[j].data.nzyxdim;
			XFLOAT *reals   = new XFLOAT[s];
			XFLOAT *imags   = new XFLOAT[s];
			XFLOAT *weights = new XFLOAT[s];

			backprojectors[j].getMdlData(reals, imags, weights);

			for (unsigned long n = 0; n < s; n++)
			{
				wsum.BPref[j].data.data[n].real   += (RFLOAT) reals[n];
				wsum.BPref[j].data.data[n].imag   += (RFLOAT) imags[n];
				wsum.BPref[j].weight.data[n]       += (RFLOAT) weights[n];
			}

			delete[] reals;
			delete[] imags;
			delete[] weights;

			backprojectors[j].clear();
		}
	}

	~MlDeviceBundle()
	{
		projectors.clear();
		backprojectors.clear();
		coarseProjectionPlans.clear();
		//Delete this lastly
		delete allocator;
	}

};
class MlOptimiserHip : public AccOptimiser
{
public:
	// transformer as holder for reuse of fftw_plans
	FourierTransformer transformer;

   //Class streams ( for concurrent scheduling of class-specific kernels)
	std::vector< hipStream_t > classStreams;
	hipStream_t defaultStream = 0;
	hipError_t errorStatus;

	HipFFT transformer1;
	HipFFT transformer2;

	MlOptimiser *baseMLO;

	bool refIs3D;
	bool dataIs3D;
        bool shiftsIs3D;

	int device_id;

	MlDeviceBundle *bundle;

	//Used for precalculations of projection setup
	HipCustomAllocator *allocator;

	//Used for precalculations of projection setup
	bool generateProjectionPlanOnTheFly;


#ifdef TIMING_FILES
	relion_timer timer;
#endif

	MlOptimiserHip(MlOptimiser *baseMLOptimiser, MlDeviceBundle* bundle, const char * timing_fnm) :
			baseMLO(baseMLOptimiser),
			transformer1(hipStreamPerThread, bundle->allocator, baseMLOptimiser->mymodel.data_dim),
			transformer2(hipStreamPerThread, bundle->allocator, baseMLOptimiser->mymodel.data_dim),
			refIs3D(baseMLO->mymodel.ref_dim == 3),
			dataIs3D(baseMLO->mymodel.data_dim == 3),
			shiftsIs3D(baseMLO->mymodel.data_dim == 3 || baseMLO->mydata.is_tomo),
                        bundle(bundle),
			device_id(bundle->device_id),
#ifdef TIMING_FILES
			timer(timing_fnm),
#endif
			errorStatus((hipError_t)0),
			allocator(bundle->allocator),
			generateProjectionPlanOnTheFly(bundle->generateProjectionPlanOnTheFly)
	{};

	void resetData() override;

	void doThreadExpectationSomeParticles(int thread_id) override;

	~MlOptimiserHip()
	{
		for (int i = 0; i < (int)classStreams.size(); i++)
			if (classStreams[i] != NULL)
				HANDLE_ERROR(hipStreamDestroy(classStreams[i]));
	}

	HipCustomAllocator *getAllocator()
	{
		return (bundle->allocator);
	};

};

#endif
