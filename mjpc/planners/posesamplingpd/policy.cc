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
        m_model = model;

        // task
        m_task = &task;

        // reference configs
        m_reference_configs.resize(m_model->nkey * m_model->nq);
    }

    // same as initialize minux framerate setting
    void PoseSamplingPDPolicy::Reset(int horizon, const double *initial_repeated_action)
    {
        mju_copy(m_reference_configs.data(), m_model->key_qpos, m_model->nkey * m_model->nq);
    }

    // set action from policy
    // TODO: Currently assumes system is fully actuated
    void PoseSamplingPDPolicy::Action(double *action, const double *state, double time) const
    {
        int offset = m_model->nq * m_task->mode;

        // Why does this alone do PD control?
        // We want target velocity 0, which is handled by joint damping of the system
        // Action automatically gets multiplied by the gain
        // Corrective gain (kp * q_current) already is copmuted by the dynamics model

        // Therefore all we need is kp * q_desired -- since kp is implicitly applied, that leaves only q_desired

        mju_copy(action, m_reference_configs.data() + offset, m_model->nu);

        // Clamp controls
        Clamp(action, m_model->actuator_ctrlrange, m_model->nu);
    }

    // copy parameters
    void PoseSamplingPDPolicy::CopyReferenceConfigsFrom(
        const vector<double> &src_reference_configs)
    {
        mju_copy(m_reference_configs.data(), src_reference_configs.data(),
                 m_model->nkey * m_model->nq);
    }

} // namespace mjpc
