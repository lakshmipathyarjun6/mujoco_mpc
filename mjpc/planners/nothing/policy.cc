#include "mjpc/planners/nothing/policy.h"

namespace mjpc
{

    // allocate memory
    void NothingPolicy::Allocate(const mjModel *model, const Task &task,
                                 int horizon)
    {
        // model
        m_model = model;

        // task
        m_task = &task;
    }

    void NothingPolicy::Reset(int horizon,
                              const double *initial_repeated_action)
    {
        // What do we reset when there is nothing to do
    }

    void NothingPolicy::Action(double *action, const double *state,
                               double time) const
    {
        // Do nothing
        mju_zero(action, m_model->nu);

        // Clamp controls
        Clamp(action, m_model->actuator_ctrlrange, m_model->nu);
    }

} // namespace mjpc
