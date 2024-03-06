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

        // special gains for ball motors
        m_ball_motor_kp = GetNumberOrDefault(1, m_model,
                                             "ball_motor_kp");
        m_ball_motor_kd = GetNumberOrDefault(1, m_model,
                                             "ball_motor_kd");
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
        // Why does this alone do PD control?
        // We want target velocity 0, which is handled by joint damping of the system
        // Action automatically gets multiplied by the gain
        // Corrective gain (kp * q_current) already is copmuted by the dynamics model

        // Therefore all we need is kp * q_desired -- since kp is implicitly applied, that leaves only q_desired
        int offset = m_model->nq * m_task->mode;
        mju_copy3(action, m_reference_configs.data() + offset);
        mju_copy(action + 6, m_reference_configs.data() + offset + 7, m_model->nu - 6);

        // EXCEPT FOR THE ROOT MOTOR ACTUATORS
        // Which we need to deal with as a special case using attitude control
        double q_desired[4];
        double q_current[4];
        double q_error[3];

        mju_copy4(q_desired, m_model->key_qpos + offset + 3);
        mju_copy4(q_current, state + 3);

        mju_subQuat(q_error, q_desired, q_current);

        // q_error = kp * ( q_desired - q )
        mju_scl3(q_error, q_error, m_ball_motor_kp);

        double w_desired[3] = {0};
        double w_current[3];
        double w_error[3];

        mju_copy3(w_current, state + m_model->nq + 3);

        // w_desired - w
        mju_sub3(w_error, w_desired, w_current);

        // w_error = kd * ( w_desired - w )
        mju_scl3(w_error, w_error, m_ball_motor_kd);

        // kp(q_error) + kd(w_error)
        double r_tau[3];

        mju_add3(r_tau, q_error, w_error);
        mju_copy3(action + 3, r_tau);

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
