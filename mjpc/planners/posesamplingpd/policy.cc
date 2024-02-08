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

        // PD gains
        m_pd_default_kp = GetNumberOrDefault(kDefaultPdKp, m_model,
                                             "default_pd_kp");
        m_pd_default_kd = GetNumberOrDefault(kDefaultPdKd, m_model,
                                             "default_pd_kd");
        m_root_pd_pos_kp = GetNumberOrDefault(kDefaultRootPosPdKp, m_model,
                                              "root_pos_pd_kp");
        m_root_pd_pos_kd = GetNumberOrDefault(kDefaultRootPosPdKd, m_model,
                                              "root_pos_pd_kd");
        m_root_pd_quat_kp = GetNumberOrDefault(kDefaultRootQuatPdKp, m_model,
                                               "root_quat_pd_kp");
        m_root_pd_quat_kd = GetNumberOrDefault(kDefaultRootQuatPdKd, m_model,
                                               "root_quat_pd_kd");
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

        double posError[kMaxSystemDofs];
        double velError[kMaxSystemDofs];

        mju_sub(posError, m_reference_configs.data() + offset, state, m_model->nu);
        mju_scl(posError, posError, m_root_pd_pos_kp, 3);
        mju_scl(posError + 3, posError + 3, m_root_pd_quat_kp, 3);
        mju_scl(posError + 6, posError + 6, m_pd_default_kp, m_model->nu - 6);

        mju_copy(velError, state + m_model->nq, m_model->nu); // want velocity close to 0
        mju_scl(velError, velError, m_root_pd_pos_kd, 3);
        mju_scl(velError + 3, velError + 3, m_root_pd_quat_kd, 3);
        mju_scl(velError + 6, velError + 6, m_pd_default_kd, m_model->nu - 6);

        mju_sub(action, posError, velError, m_model->nu);

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
