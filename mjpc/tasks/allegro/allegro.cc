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

#include "mjpc/tasks/allegro/allegro.h"

namespace mjpc
{
    // ---------- Residuals for allegro hand manipulation task ---------
    //   Number of residuals: 4
    //     Residual (0): object_position - object_traj_position
    //     Residual (1): object_orientation - object_traj_orientation
    //     Residual (2): hand joint velocity
    // ------------------------------------------------------------

    // NOTE: Currently unclear how to adapt to non-free objects (e.g. doorknob)
    void AllegroTask::ResidualFn::Residual(const mjModel *model,
                                           const mjData *data,
                                           double *residual) const
    {
        int offset = 0;

        // Query reference traj. bsplines for desired states
        vector<double> splineObjectPos = GetDesiredObjectState(data->time);

        // bool agent_object_collision_detected = false;
        vector<mjContact *> agent_object_collisions;

        // Get all collisions
        for (int c = 0; c < data->ncon; c++)
        {
            int *colliding_geoms = data->contact[c].geom; // size 2

            string colliding_geom_1 =
                model->names + model->name_geomadr[colliding_geoms[0]];
            string colliding_geom_2 =
                model->names + model->name_geomadr[colliding_geoms[1]];

            // Check for agent-object collisions only
            // Breaking into separate clauses for readability
            if (colliding_geom_1.compare(0, AGENT_GEOM_COLLIDER_PREFIX.size(),
                                         AGENT_GEOM_COLLIDER_PREFIX) == 0 &&
                colliding_geom_2.compare(0, SIM_GEOM_COLLIDER_PREFIX.size(),
                                         SIM_GEOM_COLLIDER_PREFIX) == 0)
            {
                // agent_object_collision_detected = true;
                agent_object_collisions.push_back(&data->contact[c]);
            }
            else if (colliding_geom_2.compare(
                         0, AGENT_GEOM_COLLIDER_PREFIX.size(),
                         AGENT_GEOM_COLLIDER_PREFIX) == 0 &&
                     colliding_geom_1.compare(0,
                                              SIM_GEOM_COLLIDER_PREFIX.size(),
                                              SIM_GEOM_COLLIDER_PREFIX) == 0)
            {
                // agent_object_collision_detected = true;
                agent_object_collisions.push_back(&data->contact[c]);
            }
        }

        // if (agent_object_collision_detected)
        // {
        //     cout << "Agent object collision(s) found" << endl;
        //     for (int c = 0; c < agent_object_collisions.size(); c++)
        //     {
        //         cout << "Collision #" << c << ": " << endl;
        //         cout << "\tGeometry 1: " << model->names +
        //         model->name_geomadr[agent_object_collisions[c]->geom[0]] <<
        //         endl; cout << "\tGeometry 2: " << model->names +
        //         model->name_geomadr[agent_object_collisions[c]->geom[1]] <<
        //         endl;
        //     }
        // }

        // ---------- Residual (0) ----------
        // goal position
        double goal_position[XYZ_BLOCK_SIZE];
        mju_copy3(goal_position, splineObjectPos.data());

        // system's position
        double *position = SensorByName(model, data, OBJECT_CURRENT_POSITION);

        // position error
        mju_sub3(residual + offset, position, goal_position);
        offset += 3;

        // ---------- Residual (1) ----------
        // goal orientation
        double goal_orientation[QUAT_BLOCK_SIZE];
        mju_copy4(goal_orientation, splineObjectPos.data() + 3);

        // system's orientation
        double *orientation =
            SensorByName(model, data, OBJECT_CURRENT_ORIENTATION);

        mju_normalize4(goal_orientation);
        mju_normalize4(orientation);

        // orientation error
        mju_subQuat(residual + offset, goal_orientation, orientation);
        offset += 3;

        // ---------- Residual (2) ----------
        mju_copy(residual + offset, data->qvel + 6, ALLEGRO_NON_ROOT_VEL_DOFS);
        offset += ALLEGRO_NON_ROOT_VEL_DOFS;

        // // ---------- Residual (3) ----------
        // double result[ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE];
        // mju_sub(result, m_r_contact_position_buffer,
        // m_r_contact_position_buffer + ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE,
        // ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE);

        // for(int i = 0; i < ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE; i++)
        // {
        //     result[i] *= m_r_contact_indicator_buffer[i];
        // }

        // mju_copy(residual + offset, result,
        // ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE); offset +=
        // ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE;

        // BELOW HERE IS LEGACY
        // // ---------- Residual (2) ----------
        // int handRootBodyId = mj_name2id(model, mjOBJ_BODY, ALLEGRO_ROOT);
        // int bodyJointAdr = model->body_jntadr[handRootBodyId];
        // int handQPosAdr = model->jnt_qposadr[bodyJointAdr];

        // mju_sub(residual + offset, data->qpos + handQPosAdr, m_r_qpos_buffer,
        // 3);

        // offset += 3;

        // // ---------- Residual (3) ----------
        // int rootOffset = 3;
        // mju_sub(residual + offset, data->qpos + handQPosAdr + rootOffset,
        // m_r_qpos_buffer + rootOffset, 3);

        // offset += 3;

        // // ---------- Residual (4) ----------
        // rootOffset = 6;
        // mju_sub(residual + offset, data->qpos + handQPosAdr + rootOffset,
        // m_r_qpos_buffer + rootOffset, ALLEGRO_DOFS - 6);

        // offset += ALLEGRO_DOFS - 6;

        // sensor dim sanity check
        CheckSensorDim(model, offset);
    }

    vector<double>
    AllegroTask::ResidualFn::GetDesiredAgentState(double time) const
    {
        vector<double> desiredState;

        double queryTime = fmod(time, m_bspline_loopback_time);
        double parametricTime = queryTime / m_bspline_loopback_time;

        double rootEulerAngles[XYZ_BLOCK_SIZE] = {0.0, 0.0, 0.0};
        double curveValue[2];

        for (int i = 0; i < 6; i++)
        {
            TrajectorySplineProperties *properties =
                m_hand_traj_bspline_properties[i];

            m_hand_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            double dofValue = curveValue[1];

            switch (properties->dofType)
            {
            case DofType::DOF_TYPE_ROTATION_BALL_X:
                rootEulerAngles[0] = dofValue;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Y:
                rootEulerAngles[1] = dofValue;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Z:
                rootEulerAngles[2] = dofValue;
                break;
            default:
                desiredState.push_back(dofValue);
                break;
            }
        }

        // Correct for start clamp offset
        mju_sub3(desiredState.data(), desiredState.data(),
                 m_start_clamp_offset);

        // Convert root rotation to quaternion
        double quat[QUAT_BLOCK_SIZE];
        ConvertEulerAnglesToQuat(rootEulerAngles, quat);

        for (int i = 0; i < 4; i++)
        {
            desiredState.push_back(quat[i]);
        }

        for (int i = 6; i < ALLEGRO_VEL_DOFS; i++)
        {
            m_hand_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            double dofValue = curveValue[1];

            desiredState.push_back(dofValue);
        }

        return desiredState;
    }

    vector<double>
    AllegroTask::ResidualFn::GetDesiredObjectState(double time) const
    {
        vector<double> desiredState;

        double queryTime = fmod(time, m_bspline_loopback_time);
        double parametricTime = queryTime / m_bspline_loopback_time;

        double eulerAngles[XYZ_BLOCK_SIZE] = {0.0, 0.0, 0.0};
        double curveValue[2];

        // Mopcap single body rigid object will have exactly 6 dofs
        for (int i = 0; i < 6; i++)
        {
            TrajectorySplineProperties *properties =
                m_object_traj_bspline_properties[i];

            m_object_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            double dofValue = curveValue[1];

            switch (properties->dofType)
            {
            case DofType::DOF_TYPE_ROTATION_BALL_X:
                eulerAngles[0] = dofValue;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Y:
                eulerAngles[1] = dofValue;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Z:
                eulerAngles[2] = dofValue;
                break;
            default:
                desiredState.push_back(dofValue);
                break;
            }
        }

        // Convert root rotation to quaternion
        double quat[QUAT_BLOCK_SIZE];
        ConvertEulerAnglesToQuat(eulerAngles, quat);

        for (int i = 0; i < 4; i++)
        {
            desiredState.push_back(quat[i]);
        }

        return desiredState;
    }

    // --------------------- Transition for allegro task
    // ------------------------
    //   Set `data->mocap_pos` based on `data->time` to move the object site.
    // ---------------------------------------------------------------------------
    void AllegroTask::TransitionLocked(mjModel *model, mjData *data)
    {
        // TODO: Find way to remove
        // Don't know how to turn contacts into continuous time queries
        double fps = ALLEGRO_DEFAULT_MOCAP_FPS / ALLEGRO_SLOWDOWN_FACTOR;
        double rounded_index = floor(data->time * fps);
        int contact_frame_index = int(rounded_index) % m_total_frames;

        // Reference object loading
        vector<double> splineObjectPos = GetDesiredObjectState(data->time);

        // Object mocap is first in config
        mju_copy3(data->mocap_pos, splineObjectPos.data());
        mju_copy4(data->mocap_quat, splineObjectPos.data() + XYZ_BLOCK_SIZE);

        int handPalmBodyId = mj_name2id(model, mjOBJ_BODY, ALLEGRO_MOCAP_ROOT);
        int handPalmXPosOffset = XYZ_BLOCK_SIZE * handPalmBodyId;
        int handPalmXQuatOffset = 4 * handPalmBodyId;

        int handRootBodyId = mj_name2id(model, mjOBJ_BODY, ALLEGRO_ROOT);
        int bodyJointAdr = model->body_jntadr[handRootBodyId];
        int handQPosAdr = model->jnt_qposadr[bodyJointAdr];

        // Reference hand loading
        vector<double> splineQPos = GetDesiredAgentState(data->time);

        double hand_kinematic_buffer[ALLEGRO_DOFS];

        mju_copy(hand_kinematic_buffer, data->qpos + handQPosAdr, ALLEGRO_DOFS);
        mju_copy(data->qpos + handQPosAdr, splineQPos.data(), ALLEGRO_DOFS);
        mj_kinematics(model, data);
        mju_copy(data->mocap_pos + XYZ_BLOCK_SIZE,
                 data->xpos + handPalmXPosOffset,
                 XYZ_BLOCK_SIZE * (model->nmocap - 1));
        mju_copy(data->mocap_quat + QUAT_BLOCK_SIZE,
                 data->xquat + handPalmXQuatOffset,
                 QUAT_BLOCK_SIZE * (model->nmocap - 1));
        mju_copy(data->qpos + handQPosAdr, hand_kinematic_buffer, ALLEGRO_DOFS);
        mj_kinematics(model, data);

        // Contact loading
        mju_zero(model->site_pos, ALLEGRO_MAX_CONTACTS * XYZ_BLOCK_SIZE * 2);

        // Set sameframe byte to 0 so actual offset will be computed
        for (int sid = 0; sid < model->nsite; sid++)
        {
            model->site_sameframe[sid] = 0;
        }

        int siteMetadataStartId =
            mj_name2id(model, mjOBJ_NUMERIC, SITE_DATA_START_NAME);
        int siteMetadataOffset = siteMetadataStartId + contact_frame_index;

        mjtNum *metadataData =
            model->numeric_data + model->numeric_adr[siteMetadataOffset];
        int contactDataOffset = int(metadataData[0]);
        int numActiveContacts = int(metadataData[1]);

        // Load object contact site data
        int objectContactStartSiteId =
            mj_name2id(model, mjOBJ_SITE, OBJECT_CONTACT_START_SITE_NAME);
        int objectContactDataStartId = mj_name2id(
            model, mjOBJ_NUMERIC, m_object_contact_start_data_name.c_str());

        int objectContactDataStart =
            objectContactDataStartId + contactDataOffset;

        mju_copy(model->site_pos + objectContactStartSiteId * XYZ_BLOCK_SIZE,
                 model->numeric_data +
                     model->numeric_adr[objectContactDataStart],
                 numActiveContacts * XYZ_BLOCK_SIZE);

        // Load hand contact site data
        // Doing this manually rather than running FK since reassembly and extra
        // geoms is pointlessly expensive and convoluted
        int handContactStartSiteId =
            mj_name2id(model, mjOBJ_SITE, HAND_CONTACT_START_SITE_NAME);
        int handContactDataStartId = mj_name2id(
            model, mjOBJ_NUMERIC, m_hand_contact_start_data_name.c_str());

        int handContactDataStart = handContactDataStartId + contactDataOffset;

        for (int handContactDataIndex = handContactDataStart;
             handContactDataIndex < handContactDataStart + numActiveContacts;
             handContactDataIndex++)
        {
            mjtNum *handContactDataBlock =
                model->numeric_data + model->numeric_adr[handContactDataIndex];

            int handBodyIndex = handContactDataBlock[0];

            double localCoords[XYZ_BLOCK_SIZE];
            mju_copy3(localCoords, handContactDataBlock + 1);

            int siteRelativeOffset =
                handContactDataIndex - handContactDataStart;
            int fullSiteOffset =
                (handContactStartSiteId + siteRelativeOffset) * XYZ_BLOCK_SIZE;

            mj_local2Global(data, model->site_pos + fullSiteOffset, nullptr,
                            localCoords, nullptr, handBodyIndex, 0);
        }

        mj_kinematics(model, data);

        // Reset
        if (contact_frame_index == 0)
        {
            int simObjBodyId =
                mj_name2id(model, mjOBJ_BODY, m_object_sim_body_name.c_str());
            int simObjDofs = model->nq - ALLEGRO_DOFS;

            bool objectSimBodyExists = simObjBodyId != -1;

            if (objectSimBodyExists)
            {
                int objQposadr =
                    model->jnt_qposadr[model->body_jntadr[simObjBodyId]];

                // Free joint is special since the system can't be "zeroed out"
                // due to it needing to be based off the world frame
                if (simObjDofs == 7)
                {
                    // Reset configuration to first mocap frame
                    mju_copy3(data->qpos + objQposadr, splineObjectPos.data());
                    mju_copy4(data->qpos + objQposadr + XYZ_BLOCK_SIZE,
                              splineObjectPos.data() + XYZ_BLOCK_SIZE);
                }
                else
                {
                    // Otherwise zero out the configuration
                    mju_zero(data->qvel + objQposadr, simObjDofs);
                }
            }

            mju_copy(data->qpos + handQPosAdr, splineQPos.data(), ALLEGRO_DOFS);

            // Zero out entire system velocity, acceleration, and forces
            mju_zero(data->qvel, model->nv);
            mju_zero(data->qacc, model->nv);
            mju_zero(data->ctrl, model->nu);
            mju_zero(data->actuator_force, model->nu);
            mju_zero(data->qfrc_applied, model->nv);
            mju_zero(data->xfrc_applied, model->nbody * 6);
        }
    }

    AllegroTask::AllegroTask(string objectSimBodyName,
                             string handTrajSplineFile,
                             string objectTrajSplineFile,
                             string pcHandTrajSplineFile,
                             double startClampOffsetX, double startClampOffsetY,
                             double startClampOffsetZ, int totalFrames,
                             string objectContactStartDataName,
                             string handContactStartDataName)
        : m_residual(this), m_object_sim_body_name(objectSimBodyName),
          m_total_frames(totalFrames),
          m_object_contact_start_data_name(objectContactStartDataName),
          m_hand_contact_start_data_name(handContactStartDataName)
    {
        map<string, DofType> doftypePropertyMappings;
        map<string, MeasurementUnits> measurementUnitsPropertyMappings;

        doftypePropertyMappings["rotation"] = DofType::DOF_TYPE_ROTATION;
        doftypePropertyMappings["rotationBallX"] =
            DofType::DOF_TYPE_ROTATION_BALL_X;
        doftypePropertyMappings["rotationBallY"] =
            DofType::DOF_TYPE_ROTATION_BALL_Y;
        doftypePropertyMappings["rotationBallZ"] =
            DofType::DOF_TYPE_ROTATION_BALL_Z;
        doftypePropertyMappings["translation"] = DofType::DOF_TYPE_TRANSLATION;

        measurementUnitsPropertyMappings["radians"] =
            MeasurementUnits::ROT_UNIT_RADIANS;
        measurementUnitsPropertyMappings["degrees"] =
            MeasurementUnits::ROT_UNIT_DEGREES;
        measurementUnitsPropertyMappings["meters"] =
            MeasurementUnits::TRANS_UNIT_METERS;
        measurementUnitsPropertyMappings["centimeters"] =
            MeasurementUnits::TRANS_UNIT_CENTIMETERS;
        measurementUnitsPropertyMappings["millimeters"] =
            MeasurementUnits::TRANS_UNIT_MILLIMETERS;

        m_start_clamp_offset[0] = startClampOffsetX;
        m_start_clamp_offset[1] = startClampOffsetY;
        m_start_clamp_offset[2] = startClampOffsetZ;

        mju_copy3(m_residual.m_start_clamp_offset, m_start_clamp_offset);

        Document dFullHandSplines = loadJSON(handTrajSplineFile);
        Document dObjectSplines = loadJSON(objectTrajSplineFile);
        Document dPcSplines = loadJSON(pcHandTrajSplineFile);

        m_spline_dimension = dFullHandSplines["dimension"].GetInt();
        m_spline_degree = dFullHandSplines["degree"].GetInt();
        m_spline_loopback_time = dFullHandSplines["time"].GetDouble();

        m_spline_loopback_time *= ALLEGRO_SLOWDOWN_FACTOR;

        m_residual.m_bspline_loopback_time = m_spline_loopback_time;

        m_num_pcs = dPcSplines["numComponents"].GetInt();

        for (const auto &splineData : dFullHandSplines["data"].GetArray())
        {
            int numControlPoints = splineData["numControlPoints"].GetInt();
            string dofType = splineData["type"].GetString();
            string units = splineData["units"].GetString();

            TrajectorySplineProperties *properties =
                new TrajectorySplineProperties();

            properties->numControlPoints = numControlPoints;
            properties->dofType = doftypePropertyMappings[dofType];
            properties->units = measurementUnitsPropertyMappings[units];

            vector<double> controlPoints;

            for (const auto &controlPointData :
                 splineData["controlPointData"].GetArray())
            {
                double entry = controlPointData.GetDouble();
                controlPoints.push_back(entry);
            }

            for (int i = 0; i < numControlPoints; i++)
            {
                controlPoints[i * 2] *= ALLEGRO_SLOWDOWN_FACTOR;
            }

            BSplineCurve<double> *bspc = new BSplineCurve<double>(
                m_spline_dimension, m_spline_degree, numControlPoints,
                doftypePropertyMappings[dofType],
                measurementUnitsPropertyMappings[units]);

            bspc->SetControlData(controlPoints);

            m_hand_traj_bspline_properties.push_back(properties);
            m_hand_traj_bspline_curves.push_back(bspc);

            m_residual.m_hand_traj_bspline_properties.push_back(properties);
            m_residual.m_hand_traj_bspline_curves.push_back(bspc);
        }

        if (m_hand_traj_bspline_curves.size() != ALLEGRO_VEL_DOFS)
        {
            cout << "ERROR: Expected " << ALLEGRO_VEL_DOFS << " dofs but read "
                 << m_hand_traj_bspline_curves.size() << " dofs." << endl;
        }
        else
        {
            cout << "Done" << endl;
        }

        for (const auto &splineData : dObjectSplines["data"].GetArray())
        {
            int numControlPoints = splineData["numControlPoints"].GetInt();
            string dofType = splineData["type"].GetString();
            string units = splineData["units"].GetString();

            TrajectorySplineProperties *properties =
                new TrajectorySplineProperties();

            properties->numControlPoints = numControlPoints;
            properties->dofType = doftypePropertyMappings[dofType];
            properties->units = measurementUnitsPropertyMappings[units];

            vector<double> controlPoints;

            for (const auto &controlPointData :
                 splineData["controlPointData"].GetArray())
            {
                double entry = controlPointData.GetDouble();
                controlPoints.push_back(entry);
            }

            for (int i = 0; i < numControlPoints; i++)
            {
                controlPoints[i * 2] *= ALLEGRO_SLOWDOWN_FACTOR;
            }

            BSplineCurve<double> *bspc = new BSplineCurve<double>(
                m_spline_dimension, m_spline_degree, numControlPoints,
                doftypePropertyMappings[dofType],
                measurementUnitsPropertyMappings[units]);

            bspc->SetControlData(controlPoints);

            m_object_traj_bspline_properties.push_back(properties);
            m_object_traj_bspline_curves.push_back(bspc);

            m_residual.m_object_traj_bspline_properties.push_back(properties);
            m_residual.m_object_traj_bspline_curves.push_back(bspc);
        }

        const auto pcData = dPcSplines["data"].GetObject();

        for (const auto &pcCenterData : pcData["center"].GetArray())
        {
            double dofMean = pcCenterData.GetDouble();
            m_hand_pc_center.push_back(dofMean);
        }

        for (const auto &pcSplineData : pcData["components"].GetArray())
        {
            int numControlPoints = pcSplineData["numControlPoints"].GetInt();
            string dofType = pcSplineData["type"].GetString();
            string units = pcSplineData["units"].GetString();

            TrajectorySplineProperties *properties =
                new TrajectorySplineProperties();

            properties->numControlPoints = numControlPoints;
            properties->dofType = doftypePropertyMappings[dofType];
            properties->units = measurementUnitsPropertyMappings[units];

            for (const auto &componentData :
                 pcSplineData["componentData"].GetArray())
            {
                double entry = componentData.GetDouble();
                m_hand_pc_component_matrix.push_back(entry);
            }

            vector<double> controlPoints;

            for (const auto &controlPointData :
                 pcSplineData["controlPointData"].GetArray())
            {
                double entry = controlPointData.GetDouble();
                controlPoints.push_back(entry);
            }

            for (int i = 0; i < numControlPoints; i++)
            {
                controlPoints[i * 2] *= ALLEGRO_SLOWDOWN_FACTOR;
            }

            BSplineCurve<double> *bspc = new BSplineCurve<double>(
                m_spline_dimension, m_spline_degree, numControlPoints,
                doftypePropertyMappings[dofType],
                measurementUnitsPropertyMappings[units]);

            bspc->SetControlData(controlPoints);

            m_hand_pc_traj_bspline_properties.push_back(properties);
            m_hand_pc_traj_bspline_curves.push_back(bspc);
        }

        mju_transpose(m_hand_pc_component_matrix.data(),
                      m_hand_pc_component_matrix.data(), m_num_pcs,
                      ALLEGRO_NON_ROOT_VEL_DOFS);
    }

    vector<double> AllegroTask::GetDesiredAgentState(double time) const
    {
        vector<double> desiredState;

        double queryTime = fmod(time, m_spline_loopback_time);
        double parametricTime = queryTime / m_spline_loopback_time;

        double rootEulerAngles[XYZ_BLOCK_SIZE] = {0.0, 0.0, 0.0};
        double curveValue[2];

        for (int i = 0; i < 6; i++)
        {
            TrajectorySplineProperties *properties =
                m_hand_traj_bspline_properties[i];

            m_hand_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            double dofValue = curveValue[1];

            switch (properties->dofType)
            {
            case DofType::DOF_TYPE_ROTATION_BALL_X:
                rootEulerAngles[0] = dofValue;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Y:
                rootEulerAngles[1] = dofValue;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Z:
                rootEulerAngles[2] = dofValue;
                break;
            default:
                desiredState.push_back(dofValue);
                break;
            }
        }

        // Correct for start clamp offset
        mju_sub3(desiredState.data(), desiredState.data(),
                 m_start_clamp_offset);

        // Convert root rotation to quaternion
        double quat[QUAT_BLOCK_SIZE];
        ConvertEulerAnglesToQuat(rootEulerAngles, quat);

        for (int i = 0; i < 4; i++)
        {
            desiredState.push_back(quat[i]);
        }

        for (int i = 6; i < ALLEGRO_VEL_DOFS; i++)
        {
            m_hand_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            double dofValue = curveValue[1];

            desiredState.push_back(dofValue);
        }

        return desiredState;
    }

    vector<double> AllegroTask::GetDesiredAgentStateFromPCs(double time) const
    {
        vector<double> desiredState;

        double queryTime = fmod(time, m_spline_loopback_time);
        double parametricTime = queryTime / m_spline_loopback_time;

        // Get root state from original spline - no PC is performed on it
        double rootEulerAngles[XYZ_BLOCK_SIZE] = {0.0, 0.0, 0.0};
        double curveValue[2];

        for (int i = 0; i < 6; i++)
        {
            TrajectorySplineProperties *properties =
                m_hand_traj_bspline_properties[i];

            m_hand_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            double dofValue = curveValue[1];

            switch (properties->dofType)
            {
            case DofType::DOF_TYPE_ROTATION_BALL_X:
                rootEulerAngles[0] = dofValue;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Y:
                rootEulerAngles[1] = dofValue;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Z:
                rootEulerAngles[2] = dofValue;
                break;
            default:
                desiredState.push_back(dofValue);
                break;
            }
        }

        // Correct for start clamp offset
        mju_sub3(desiredState.data(), desiredState.data(),
                 m_start_clamp_offset);

        // Convert root rotation to quaternion
        double quat[QUAT_BLOCK_SIZE];
        ConvertEulerAnglesToQuat(rootEulerAngles, quat);

        for (int i = 0; i < 4; i++)
        {
            desiredState.push_back(quat[i]);
        }

        vector<double> pcState;
        pcState.resize(m_num_pcs);

        for (int i = 0; i < m_num_pcs; i++)
        {
            m_hand_pc_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            pcState[i] = curveValue[1];
        }

        vector<double> uncompressedState;
        uncompressedState.resize(ALLEGRO_NON_ROOT_VEL_DOFS);

        mju_mulMatVec(uncompressedState.data(),
                      m_hand_pc_component_matrix.data(), pcState.data(),
                      ALLEGRO_NON_ROOT_VEL_DOFS, m_num_pcs);

        mju_addTo(uncompressedState.data(), m_hand_pc_center.data(),
                  ALLEGRO_NON_ROOT_VEL_DOFS);

        for (int i = 0; i < ALLEGRO_NON_ROOT_VEL_DOFS; i++)
        {
            double dofValue = uncompressedState[i];
            desiredState.push_back(dofValue);
        }

        return desiredState;
    }

    vector<double> AllegroTask::GetDesiredObjectState(double time) const
    {
        vector<double> desiredState;

        double queryTime = fmod(time, m_spline_loopback_time);
        double parametricTime = queryTime / m_spline_loopback_time;

        double eulerAngles[XYZ_BLOCK_SIZE] = {0.0, 0.0, 0.0};
        double curveValue[2];

        // Mopcap single body rigid object will have exactly 6 dofs
        for (int i = 0; i < 6; i++)
        {
            TrajectorySplineProperties *properties =
                m_object_traj_bspline_properties[i];

            m_object_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            double dofValue = curveValue[1];

            switch (properties->dofType)
            {
            case DofType::DOF_TYPE_ROTATION_BALL_X:
                eulerAngles[0] = dofValue;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Y:
                eulerAngles[1] = dofValue;
                break;
            case DofType::DOF_TYPE_ROTATION_BALL_Z:
                eulerAngles[2] = dofValue;
                break;
            default:
                desiredState.push_back(dofValue);
                break;
            }
        }

        // Convert root rotation to quaternion
        double quat[QUAT_BLOCK_SIZE];
        ConvertEulerAnglesToQuat(eulerAngles, quat);

        for (int i = 0; i < 4; i++)
        {
            desiredState.push_back(quat[i]);
        }

        return desiredState;
    }

    vector<vector<double>> AllegroTask::GetAgentBSplineControlData(
        int &dimension, int &degree, double &loopbackTime,
        double translationOffset[XYZ_BLOCK_SIZE], vector<DofType> &dofTypes,
        vector<MeasurementUnits> &measurementUnits) const
    {
        vector<vector<double>> bsplineControlData;

        dimension = m_spline_dimension;
        degree = m_spline_degree;
        loopbackTime = m_spline_loopback_time;

        mju_copy3(translationOffset, m_start_clamp_offset);

        dofTypes.clear();
        measurementUnits.clear();

        int numSplines = m_hand_traj_bspline_curves.size();

        for (int i = 0; i < numSplines; i++)
        {
            vector<double> dataCopy; // Deep copy to avoid modifying original

            vector<double> dataOriginal =
                m_hand_traj_bspline_curves[i]->GetControlData();

            TrajectorySplineProperties *properties =
                m_hand_traj_bspline_properties[i];

            int numElements = dataOriginal.size();

            // Intentionally avoid memcpy
            for (int j = 0; j < numElements; j++)
            {
                dataCopy.push_back(dataOriginal[j]);
            }

            bsplineControlData.push_back(dataCopy);
            dofTypes.push_back(properties->dofType);
            measurementUnits.push_back(properties->units);
        }

        return bsplineControlData;
    }

    vector<vector<double>> AllegroTask::GetAgentPCBSplineControlData(
        int &dimension, int &degree, double &loopbackTime, int &numMaxPCs,
        vector<double> &centerData, vector<double> &componentData,
        double translationOffset[XYZ_BLOCK_SIZE], vector<DofType> &dofTypes,
        vector<MeasurementUnits> &measurementUnits) const
    {
        vector<vector<double>> bsplineControlData;

        dimension = m_spline_dimension;
        degree = m_spline_degree;
        loopbackTime = m_spline_loopback_time;
        numMaxPCs = m_num_pcs;

        mju_copy3(translationOffset, m_start_clamp_offset);

        dofTypes.clear();
        measurementUnits.clear();
        centerData.clear();
        componentData.clear();

        for (int i = 0; i < 6; i++)
        {
            vector<double> dataCopy; // Deep copy to avoid modifying original

            vector<double> dataOriginal =
                m_hand_traj_bspline_curves[i]->GetControlData();

            TrajectorySplineProperties *properties =
                m_hand_traj_bspline_properties[i];

            int numElements = dataOriginal.size();

            // Intentionally avoid memcpy
            for (int j = 0; j < numElements; j++)
            {
                dataCopy.push_back(dataOriginal[j]);
            }

            bsplineControlData.push_back(dataCopy);
            dofTypes.push_back(properties->dofType);
            measurementUnits.push_back(properties->units);
        }

        for (int i = 0; i < m_num_pcs; i++)
        {
            vector<double> dataCopy; // Deep copy to avoid modifying original

            vector<double> dataOriginal =
                m_hand_pc_traj_bspline_curves[i]->GetControlData();

            TrajectorySplineProperties *properties =
                m_hand_pc_traj_bspline_properties[i];

            int numElements = dataOriginal.size();

            // Intentionally avoid memcpy
            for (int j = 0; j < numElements; j++)
            {
                dataCopy.push_back(dataOriginal[j]);
            }

            bsplineControlData.push_back(dataCopy);
            dofTypes.push_back(properties->dofType);
            measurementUnits.push_back(properties->units);
        }

        for (int i = 0; i < m_hand_pc_center.size(); i++)
        {
            centerData.push_back(m_hand_pc_center[i]);
        }

        for (int i = 0; i < m_hand_pc_component_matrix.size(); i++)
        {
            componentData.push_back(m_hand_pc_component_matrix[i]);
        }

        return bsplineControlData;
    }

    string AllegroApplePassTask::XmlPath() const
    {
        return GetModelPath("allegro/task_apple_pass.xml");
    }

    string AllegroApplePassTask::Name() const { return "Allegro Apple Pass"; }

    string AllegroDoorknobUseTask::XmlPath() const
    {
        return GetModelPath("allegro/task_doorknob_use.xml");
    }

    string AllegroDoorknobUseTask::Name() const
    {
        return "Allegro Doorknob Use";
    }

} // namespace mjpc
