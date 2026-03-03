#ifndef ACC_OPTIMISER_H_
#define ACC_OPTIMISER_H_

class AccOptimiser {
public:
    virtual void resetData() = 0;
    virtual void doThreadExpectationSomeParticles(int thread_id) = 0;
    virtual ~AccOptimiser() = default;
};

#endif /* ACC_OPTIMISER_H_ */
