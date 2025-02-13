#include "mjpc/planners/pcbsplinesampling/policy.h"

namespace mjpc
{

    // allocate memory
    void PCBSplineSamplingPolicy::Allocate(const mjModel *model,
                                           const Task &task, int horizon)
    {
        // model
        m_model = model;

        // task
        m_task = &task;

        // original bspline data
        // direct copy-pasta from planner solely to preserve inheritance
        // structure
        m_bspline_control_data = m_task->GetAgentPCBSplineControlData(
            m_bspline_dimension, m_bspline_degree, m_bspline_loopback_time,
            m_num_max_pcs, m_pc_center, m_pc_component_matrix,
            m_bspline_translation_offset, m_bspline_doftype_data,
            m_bspline_measurementunit_data);

        // sanity checks
        // currently assumes agents are fully actuated
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

        m_num_agent_joints =
            m_model->njnt - 1; // last joint is simulated object
        m_agent_joints.resize(m_num_agent_joints);

        for (int i = 0; i < m_num_agent_joints; i++)
        {
            m_agent_joints[i] = m_model->jnt_type[i];
        }

        // generate bsplines
        GenerateBSplineControlData();

        // special gains for ball motors
        m_root_ball_motor_kp =
            GetNumberOrDefault(1, m_model, "root_ball_motor_kp");
        m_root_ball_motor_kd =
            GetNumberOrDefault(1, m_model, "root_ball_motor_kd");

        m_intermediate_ball_motor_kp =
            GetNumberOrDefault(1, m_model, "intermediate_ball_motor_kp");
        m_intermediate_ball_motor_kd =
            GetNumberOrDefault(1, m_model, "intermediate_ball_motor_kd");

        AdjustPCComponentMatrix(m_num_max_pcs);
    }

    void PCBSplineSamplingPolicy::Reset(int horizon,
                                        const double *initial_repeated_action)
    {
        m_bspline_control_data.clear();
        m_bspline_doftype_data.clear();
        m_bspline_measurementunit_data.clear();

        // original bspline data
        m_bspline_control_data = m_task->GetAgentPCBSplineControlData(
            m_bspline_dimension, m_bspline_degree, m_bspline_loopback_time,
            m_num_max_pcs, m_pc_center, m_pc_component_matrix,
            m_bspline_translation_offset, m_bspline_doftype_data,
            m_bspline_measurementunit_data);

        GenerateBSplineControlData();
    }

    void PCBSplineSamplingPolicy::Action(double *action, const double *state,
                                         double time) const
    {
        // Do pc bspline sampling
        vector<double> desiredState = ComputeDesiredAgentState(time);

        // For now, assume translation and hinge joints are servo controlled
        // while attitude is motor controlled

        int posDofIndex = 0;
        int velDofIndex = 0;

        for (int jointIndex = 0; jointIndex < m_num_agent_joints; jointIndex++)
        {
            switch (m_agent_joints[jointIndex])
            {
            case mjJNT_BALL:
            {
                // Use attitude control for 3-motor actuated ball joints
                double q_desired[4];
                double q_current[4];
                double q_error[3];

                mju_copy4(q_desired, desiredState.data() + posDofIndex);
                mju_copy4(q_current, state + posDofIndex);

                mju_subQuat(q_error, q_desired, q_current);

                // q_error = kp * ( q_desired - q )
                double kp = (jointIndex > 3) ? m_intermediate_ball_motor_kp
                                             : m_root_ball_motor_kp;
                mju_scl3(q_error, q_error, kp);

                double w_desired[3] = {0};
                double w_current[3];
                double w_error[3];

                mju_copy3(w_current, state + m_model->nq + velDofIndex);

                // w_desired - w
                mju_sub3(w_error, w_desired, w_current);

                // w_error = kd * ( w_desired - w )
                double kd = (jointIndex > 3) ? m_intermediate_ball_motor_kd
                                             : m_root_ball_motor_kd;
                mju_scl3(w_error, w_error, kd);

                // kp(q_error) + kd(w_error)
                double r_tau[3];

                mju_add3(r_tau, q_error, w_error);
                mju_copy3(action + velDofIndex, r_tau);

                posDofIndex += 4;
                velDofIndex += 3;
            }
            break;
            case mjJNT_SLIDE:
            case mjJNT_HINGE:
            {
                // Why does this alone do PD control?
                // We want target velocity 0, which is handled by joint damping
                // of the system Action automatically gets multiplied by the
                // gain Corrective gain (kp * q_current) already is copmuted by
                // the dynamics model

                // Therefore all we need is kp * q_desired -- since kp is
                // implicitly
                // applied, that leaves only q_desired

                mju_copy(action + velDofIndex,
                         desiredState.data() + posDofIndex, 1);

                posDofIndex += 1;
                velDofIndex += 1;
            }
            break;
            default:
            {
                cout << "Unsupported joint type" << endl;
                return;
            }
            break;
            }
        }

        // Clamp controls
        Clamp(action, m_model->actuator_ctrlrange, m_model->nu);
    }

    // Reduce the number of components to the number of active pcs being
    // used
    void PCBSplineSamplingPolicy::AdjustPCComponentMatrix(int numActivePCs)
    {
        m_active_pc_component_matrix.clear();

        m_num_active_pcs = numActivePCs;

        int numNonRootDofs = m_model->nu - 6;
        m_active_pc_component_matrix.resize(m_num_active_pcs * numNonRootDofs);

        vector<double> componentMatrixBuff;
        componentMatrixBuff.resize(m_num_max_pcs * numNonRootDofs);

        mju_transpose(componentMatrixBuff.data(), m_pc_component_matrix.data(),
                      numNonRootDofs, m_num_active_pcs);

        mju_copy(m_active_pc_component_matrix.data(),
                 componentMatrixBuff.data(), m_num_active_pcs * numNonRootDofs);

        mju_transpose(m_active_pc_component_matrix.data(),
                      m_active_pc_component_matrix.data(), m_num_active_pcs,
                      numNonRootDofs);
    }

    // copy bspline control points
    void PCBSplineSamplingPolicy::CopyControlPointsAndActivePCsFrom(
        const PCBSplineSamplingPolicy &policy)
    {
        m_bspline_control_data.clear();

        copy(policy.m_bspline_control_data.begin(),
             policy.m_bspline_control_data.end(),
             back_inserter(m_bspline_control_data));

        m_num_active_pcs = policy.m_num_active_pcs;

        GenerateBSplineControlData();
    }

    // deltas should be of size m_model->nu * m_num_bspline_control_points *
    // m_bspline_dimension
    void PCBSplineSamplingPolicy::AdjustBSplineControlPoints(double *deltas)
    {
        for (int rootDofIndex = 0; rootDofIndex < 6; rootDofIndex++)
        {
            mju_addTo(m_bspline_control_data[rootDofIndex].data(),
                      deltas + rootDofIndex * m_num_bspline_control_points *
                                   m_bspline_dimension,
                      m_num_bspline_control_points * m_bspline_dimension);

            m_control_bspline_curves[rootDofIndex]->SetControlData(
                m_bspline_control_data[rootDofIndex]);
        }

        // Ignore noise for all inactive PCs
        for (int pcDofIndex = 0; pcDofIndex < m_num_active_pcs; pcDofIndex++)
        {
            int dofOffset = 6 + pcDofIndex;

            mju_addTo(m_bspline_control_data[dofOffset].data(),
                      deltas + pcDofIndex * m_num_bspline_control_points *
                                   m_bspline_dimension,
                      m_num_bspline_control_points * m_bspline_dimension);

            m_control_bspline_curves[dofOffset]->SetControlData(
                m_bspline_control_data[dofOffset]);
        }
    }

    // Begin private methods

    vector<double>
    PCBSplineSamplingPolicy::ComputeDesiredAgentState(double time) const
    {
        vector<double> desiredState;

        double queryTime = fmod(time, m_bspline_loopback_time);
        double parametricTime = queryTime / m_bspline_loopback_time;

        double position, velocity;

        vector<double> rawSplinePositions;
        rawSplinePositions.resize(m_model->nu);

        vector<double> rawSplineVelocities;
        rawSplineVelocities.resize(m_model->nu);

        vector<double> pcSplinePositions;
        pcSplinePositions.resize(m_num_active_pcs);

        vector<double> pcSplineVelocities;
        pcSplineVelocities.resize(m_num_active_pcs);

        for (int i = 0; i < 6; i++)
        {
            m_control_bspline_curves[i]
                ->GetPositionAndVelocityInMeasurementUnits(parametricTime,
                                                           position, velocity);

            rawSplinePositions[i] = position;
            rawSplineVelocities[i] = velocity;
        }

        for (int i = 0; i < m_num_active_pcs; i++)
        {
            m_control_bspline_curves[i + 6]
                ->GetPositionAndVelocityInMeasurementUnits(parametricTime,
                                                           position, velocity);

            pcSplinePositions[i] = position;
            pcSplineVelocities[i] = velocity;
        }

        vector<double> uncompressedState;
        int numNonRootDofs = m_model->nu - 6;

        uncompressedState.resize(numNonRootDofs);

        mju_mulMatVec(
            uncompressedState.data(), m_active_pc_component_matrix.data(),
            pcSplinePositions.data(), numNonRootDofs, m_num_active_pcs);

        mju_addTo(uncompressedState.data(), m_pc_center.data(), numNonRootDofs);

        for (int i = 0; i < numNonRootDofs; i++)
        {
            rawSplinePositions[i + 6] = uncompressedState[i];
        }

        int dofIndex = 0;
        int jointIndex = 0;

        while (jointIndex < m_num_agent_joints)
        {
            int readSize = 0;

            switch (m_agent_joints[jointIndex])
            {
            case mjJNT_BALL:
            {
                readSize = 3;
                jointIndex += 1;

                double ballAngles[3];
                mju_copy3(ballAngles, rawSplinePositions.data() + dofIndex);

                // Convert to quaternion
                double quat[4];
                ConvertEulerAnglesToQuat(ballAngles, quat);

                for (int j = 0; j < 4; j++)
                {
                    desiredState.push_back(quat[j]);
                }
            }
            break;
            case mjJNT_SLIDE:
            {
                // AKA: If the root joint is a translational dof
                if (jointIndex == 0 && m_num_agent_joints > 2 &&
                    m_agent_joints[jointIndex + 1] == mjJNT_SLIDE &&
                    m_agent_joints[jointIndex + 2] == mjJNT_SLIDE)
                {
                    readSize = 3;
                    jointIndex += 3;

                    double transDof[3];
                    mju_copy3(transDof, rawSplinePositions.data() + dofIndex);

                    // Correct for start clamp offset
                    mju_sub3(transDof, transDof, m_bspline_translation_offset);

                    for (int j = 0; j < 3; j++)
                    {
                        desiredState.push_back(transDof[j]);
                    }
                }
                else
                {
                    readSize = 1;
                    desiredState.push_back(rawSplinePositions[dofIndex]);
                }
            }
            break;
            case mjJNT_HINGE:
            {
                readSize = 1;
                jointIndex += 1;

                desiredState.push_back(rawSplinePositions[dofIndex]);
            }
            break;
            default:
            {
                cout << "Unsupported joint type" << endl;
                return desiredState;
            }
            break;
            }

            dofIndex += readSize;
        }

        return desiredState;
    }

    void PCBSplineSamplingPolicy::GenerateBSplineControlData()
    {
        int numControlSplinesToGenerate = m_num_max_pcs + 6;

        m_control_bspline_curves.clear();

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
