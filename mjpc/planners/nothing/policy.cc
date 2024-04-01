// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
