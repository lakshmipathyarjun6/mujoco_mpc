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

#include "mjpc/planners/posesamplingpd/policy.h"

namespace mjpc
{

    // allocate memory
    void PoseSamplingPDPolicy::Allocate(const mjModel *model, const Task &task,
                                        int horizon)
    {
        // model
        this->model = model;
    }

    // reset memory to zeros
    void PoseSamplingPDPolicy::Reset(int horizon, const double *initial_repeated_action)
    {
    }

    // set action from policy
    void PoseSamplingPDPolicy::Action(double *action, const double *state, double time) const
    {
        double rounded_index = floor(time * FPS);
        int current_index = int(rounded_index) % model->nkey;

        int handMocapQOffset = model->nq * current_index;

        double posError[MaxDOFs];
        double velError[MaxDOFs];

        mju_sub(posError, model->key_qpos + handMocapQOffset, state, model->nu);
        mju_scl(posError, posError, 20, 3);
        mju_scl(posError + 3, posError + 3, 10, 3);
        mju_scl(posError + 6, posError + 6, 5, model->nu - 6);

        mju_copy(velError, state + model->nq, model->nu); // want velocity close to 0
        mju_scl(velError, velError, 1, 3);
        mju_scl(velError + 3, velError + 3, 0, model->nu - 3);

        mju_sub(action, posError, velError, model->nu);

        // Clamp controls
        Clamp(action, model->actuator_ctrlrange, model->nu);
    }

} // namespace mjpc
