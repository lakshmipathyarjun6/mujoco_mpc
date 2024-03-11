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

        // original bspline data
        m_bspline_control_data = m_task->GetBSplineControlData(
            m_bspline_dimension, m_bspline_degree, m_bspline_loopback_time,
            m_bspline_translation_offset, m_bspline_doftype_data,
            m_bspline_measurementunit_data);

        // sanity checks
        if (m_bspline_control_data.size() > 0 &&
            m_model->nu != m_bspline_control_data.size())
        {
            cout << "ERROR: Number of BSpline curves does not match number of "
                    "DOFs!"
                 << endl;
            return;
        }
        if (m_bspline_control_data.size() > 0)
        {
            m_num_bspline_control_points =
                m_bspline_control_data[0].size() / m_bspline_dimension;

            for (int i = 0; i < m_model->nu; i++)
            {
                if (m_bspline_control_data[i].size() / m_bspline_dimension !=
                    m_num_bspline_control_points)
                {
                    cout << "ERROR: BSplines must have same number of control "
                            "points."
                         << endl;
                    return;
                }
            }

            if (m_bspline_doftype_data.size() != m_model->nu)
            {
                cout << "ERROR: Each bspline must have a dof type." << endl;
                return;
            }
            if (m_bspline_measurementunit_data.size() != m_model->nu)
            {
                cout << "ERROR: Each bspline must have a measurement unit."
                     << endl;
                return;
            }
        }

        // generate bsplines
        GenerateBSplineControlData();

        // reference configs
        m_reference_configs.resize(m_model->nkey * m_model->nq);

        // special gains for ball motors
        m_ball_motor_kp = GetNumberOrDefault(1, m_model, "ball_motor_kp");
        m_ball_motor_kd = GetNumberOrDefault(1, m_model, "ball_motor_kd");
    }

    // same as initialize minux framerate setting
    void PoseSamplingPDPolicy::Reset(int horizon,
                                     const double *initial_repeated_action)
    {
        mju_copy(m_reference_configs.data(), m_model->key_qpos,
                 m_model->nkey * m_model->nq);
    }

    // set action from policy
    // TODO: Currently assumes system is fully actuated
    void PoseSamplingPDPolicy::Action(double *action, const double *state,
                                      double time) const
    {
        // Compute desired qpos from bsplines
        vector<double> spline_qpos;

        double query_time = fmod(time, m_bspline_loopback_time);
        double parametric_time = query_time / m_bspline_loopback_time;

        vector<double> curve_value;
        curve_value.resize(m_bspline_dimension);

        // TODO: Currently assumes that root is a trans + ball joint
        double ball_euler_angles[3] = {0.0, 0.0, 0.0};

        for (int i = 0; i < 6; i++)
        {
            m_control_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametric_time, curve_value.data());

            double dof_value = curve_value[1];

            switch (m_bspline_doftype_data[i])
            {
            case DofType::DOF_TYPE_ROTATION_BALL_X:
                ball_euler_angles[0] = dof_value;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Y:
                ball_euler_angles[1] = dof_value;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Z:
                ball_euler_angles[2] = dof_value;
                break;
            default:
                spline_qpos.push_back(dof_value);
                break;
            }
        }

        // Correct for start clamp offset
        mju_sub3(spline_qpos.data(), spline_qpos.data(),
                 m_bspline_translation_offset);

        // Convert root rotation to quaternion
        double quat[4];
        ConvertEulerAnglesToQuat(ball_euler_angles, quat);

        for (int i = 0; i < 4; i++)
        {
            spline_qpos.push_back(quat[i]);
        }

        for (int i = 6; i < m_model->nu; i++)
        {
            m_control_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametric_time, curve_value.data());

            double dof_value = curve_value[1];

            spline_qpos.push_back(dof_value);
        }

        // Why does this alone do PD control?
        // We want target velocity 0, which is handled by joint damping of the
        // system Action automatically gets multiplied by the gain Corrective
        // gain (kp * q_current) already is copmuted by the dynamics model

        // Therefore all we need is kp * q_desired -- since kp is implicitly
        // applied, that leaves only q_desired
        mju_copy3(action, spline_qpos.data());
        mju_copy(action + 6, spline_qpos.data() + 7, m_model->nu - 6);

        // EXCEPT FOR THE ROOT MOTOR ACTUATORS
        // Which we need to deal with as a special case using attitude control
        double q_desired[4];
        double q_current[4];
        double q_error[3];

        mju_copy4(q_desired, spline_qpos.data() + 3);
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

    void PoseSamplingPDPolicy::GenerateBSplineControlData()
    {
        int numControlSplinesToGenerate = m_model->nu;

        if (m_control_bspline_curves.size() > 0)
        {
            for (int i = 0; i < m_control_bspline_curves.size(); i++)
            {
                delete m_control_bspline_curves[i];
            }

            m_control_bspline_curves.clear();
        }

        for (int i = 0; i < numControlSplinesToGenerate; i++)
        {
            vector<double> splineControlPointData = m_bspline_control_data[i];

            BSplineCurve<double> *bspc = new BSplineCurve<double>(
                m_bspline_dimension, m_bspline_degree,
                m_num_bspline_control_points, m_bspline_doftype_data[i],
                m_bspline_measurementunit_data[i]);

            bspc->SetControlData(splineControlPointData);

            m_control_bspline_curves.push_back(bspc);
        }
    }

} // namespace mjpc
