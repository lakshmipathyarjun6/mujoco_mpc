#ifndef MJPC_PLANNERS_NOTHING_POLICY_H_
#define MJPC_PLANNERS_NOTHING_POLICY_H_

#include <mujoco/mujoco.h>

#include "mjpc/planners/policy.h"
#include "mjpc/utilities.h"

using namespace std;

namespace mjpc
{

    // the policy that does nothing
    class NothingPolicy : public Policy
    {
    public:
        // constructor
        NothingPolicy() = default;

        // destructor
        ~NothingPolicy() override = default;

        // ----- methods ----- //

        // allocate memory
        void Allocate(const mjModel *model, const Task &task,
                      int horizon) override;

        // reset memory to zeros and reference configs to original mocap
        // trajectory
        void Reset(int horizon,
                   const double *initial_repeated_action = nullptr) override;

        // set action from policy
        void Action(double *action, const double *state,
                    double time) const override;

    private:
        // ----- members ----- //
        const mjModel *m_model;
        const Task *m_task;
    };

} // namespace mjpc

#endif // MJPC_PLANNERS_NOTHING_POLICY_H_
