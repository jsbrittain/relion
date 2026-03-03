// For the SYCL version, this is essentially a mix of
// cuda_ml_optimiser.h and cpu_ml_optimiser.h.
// Note the the SYCL implementation defines the floating point precision used
// for XFLOAT using ACC_DOUBLE_PRECISION (ACC_DOUBLE_PRECISION is also used
// for the equivalent purpose throughout the code)
#ifndef SYCL_ML_OPTIMISER_H_
#define SYCL_ML_OPTIMISER_H_

#include <stack>
#include <vector>
#include <tuple>
#include <typeinfo>

#include "src/mpi.h"
#include "src/ml_optimiser.h"
#include "src/acc/acc_projector_plan.h"
#include "src/acc/acc_projector.h"
#include "src/acc/acc_backprojector.h"
#include "src/acc/sycl/mkl_fft.h"
#include "src/acc/sycl/sycl_benchmark_utils.h"
#include "src/acc/acc_bundle.h"
#include "src/acc/acc_optimiser.h"

#include "src/acc/acc_ml_optimiser.h"
#include "src/acc/acc_ptr.h"

#include "src/acc/sycl/sycl_virtual_dev.h"

class MlSyclDataBundle : public AccBundle
{
public:
	void setup(MlOptimiser *baseMLO);
	void syncAllBackprojects() override	{ _devAcc->waitAll(); }
	virtualSYCL* getSyclDevice()	{ return _devAcc; }

	void extractAndAccumulate(MlWsumModel &wsum) override
	{
		for (int j = 0; j < (int)backprojectors.size(); j++)
		{
			unsigned long s = wsum.BPref[j].data.nzyxdim;
			deviceStream_t stream = backprojectors[j].stream;
			XFLOAT *reals   = (XFLOAT*)(stream->syclMalloc(s * sizeof(XFLOAT), syclMallocType::host));
			XFLOAT *imags   = (XFLOAT*)(stream->syclMalloc(s * sizeof(XFLOAT), syclMallocType::host));
			XFLOAT *weights = (XFLOAT*)(stream->syclMalloc(s * sizeof(XFLOAT), syclMallocType::host));

			backprojectors[j].getMdlData(reals, imags, weights);

			for (unsigned long n = 0; n < s; n++)
			{
				wsum.BPref[j].data.data[n].real   += (RFLOAT) reals[n];
				wsum.BPref[j].data.data[n].imag   += (RFLOAT) imags[n];
				wsum.BPref[j].weight.data[n]       += (RFLOAT) weights[n];
			}

			stream->syclFree(reals);
			stream->syclFree(imags);
			stream->syclFree(weights);

			backprojectors[j].clear();
		}
	}

	MlSyclDataBundle(virtualSYCL *dev);
	~MlSyclDataBundle();

private:
	virtualSYCL *_devAcc;
};

class MlOptimiserSYCL : public AccOptimiser
{
public:
	MlOptimiser *baseMLO;
	MlSyclDataBundle *bundle;

	// transformer as holder for reuse of fftw_plans
	FourierTransformer transformer;

	MklFFT transformer1;
	MklFFT transformer2;

	bool refIs3D;
	bool dataIs3D;
	bool shiftsIs3D;

	int threadID;

	std::vector<virtualSYCL*> classStreams;

	static void checkDevices();
	static std::vector<virtualSYCL*> getDevices(const syclDeviceType select, const std::tuple<bool,bool,bool> syclOpt, const syclBackendType BE = syclBackendType::levelZero, const bool verbose = true);

	virtualSYCL* getSyclDevice()	{ return _devAcc; }
	bool useStream() const	{ return _useStream; }

	//Used for precalculations of projection setup
	bool generateProjectionPlanOnTheFly;

	void setupDevice();

	void resetData() override;

	void expectationOneParticle(unsigned long my_part_id, const int thread_id);
	void doThreadExpectationSomeParticles(const int thread_id) override;

	void* getAllocator()
	{
		return nullptr;
	};

	MlOptimiserSYCL(MlOptimiser *baseMLOptimiser, MlSyclDataBundle *b, const bool isStream, const char *timing_fnm);

	~MlOptimiserSYCL();

private:
	virtualSYCL *_devAcc;
	bool _useStream;
};
#endif
