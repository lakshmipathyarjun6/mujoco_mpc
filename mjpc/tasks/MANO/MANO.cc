#include "mjpc/tasks/MANO/MANO.h"

namespace mjpc
{
    // ---------- Residuals for MANO hand manipulation task ---------
    //   Number of residuals: 4
    //     Residual (0): object_position - object_traj_position
    //     Residual (1): object_orientation - object_traj_orientation
    // ------------------------------------------------------------

    // NOTE: Currently unclear how to adapt to non-free objects (e.g. doorknob)
    void MANOTask::ResidualFn::Residual(const mjModel *model,
                                        const mjData *data,
                                        double *residual) const
    {
        int offset = 0;

        // ---------- Residual (0) ----------
        // goal position
        double goal_position[3];
        mju_copy3(goal_position, m_r_object_mocap_pos_buffer);

        // system's position
        double *position = SensorByName(model, data, OBJECT_CURRENT_POSITION);

        // position error
        mju_sub3(residual + offset, position, goal_position);
        offset += 3;

        // ---------- Residual (1) ----------
        // goal orientation
        double goal_orientation[4];
        mju_copy4(goal_orientation, m_r_object_mocap_quat_buffer);

        // system's orientation
        double *orientation =
            SensorByName(model, data, OBJECT_CURRENT_ORIENTATION);

        mju_normalize4(goal_orientation);
        mju_normalize4(orientation);

        // orientation error
        mju_subQuat(residual + offset, goal_orientation, orientation);
        offset += 3;

        // ---------- Residual (2) ----------
        double full_result[MAX_CONTACTS * XYZ_BLOCK_SIZE];
        double relevant_result[MAX_CONTACTS * XYZ_BLOCK_SIZE];

        fill(begin(full_result), end(full_result), 0);
        fill(begin(relevant_result), end(relevant_result), 0);

        mju_sub(full_result, m_r_hand_contact_position_buffer,
                m_r_object_contact_position_buffer,
                MAX_CONTACTS * XYZ_BLOCK_SIZE);

        for (int i = 0; i < m_num_active_contacts; i++)
        {
            relevant_result[i] = full_result[i];
        }

        mju_copy(residual + offset, relevant_result,
                 MAX_CONTACTS * XYZ_BLOCK_SIZE);
        offset += MAX_CONTACTS * XYZ_BLOCK_SIZE;

        // sensor dim sanity check
        CheckSensorDim(model, offset);
    }

    // --------------------- Transition for MANO task
    // ------------------------
    //   Set `data->mocap_pos` based on `data->time` to move the object site.
    // ---------------------------------------------------------------------------
    void MANOTask::TransitionLocked(mjModel *model, mjData *data)
    {
        // TODO: Find way to remove
        // Don't know how to turn contacts into continuous time queries
        double fps = DEFAULT_MOCAP_FPS / SLOWDOWN_FACTOR;
        double rounded_index = floor(data->time * fps);
        int contact_frame_index = int(rounded_index) % m_total_frames;

        // Reference object loading
        vector<double> splineObjectPos = GetDesiredObjectState(data->time);

        mju_copy3(m_residual.m_r_object_mocap_pos_buffer,
                  splineObjectPos.data());
        mju_copy4(m_residual.m_r_object_mocap_quat_buffer,
                  splineObjectPos.data() + 3);

        // Object mocap is first in config
        mju_copy3(data->mocap_pos, splineObjectPos.data());
        mju_copy4(data->mocap_quat, splineObjectPos.data() + 3);

        // Fetch hand data
        int handRootBodyId = mj_name2id(model, mjOBJ_BODY, MANO_ROOT);
        int bodyJointAdr = model->body_jntadr[handRootBodyId];
        int handQPosAdr = model->jnt_qposadr[bodyJointAdr];

        int handWristXPosOffset = 3 * handRootBodyId;
        int handWristXQuatOffset = 4 * handRootBodyId;

        // Reference hand loading
        vector<double> splineQPos = GetDesiredAgentState(data->time);

        mju_copy(m_hand_kinematic_buffer, data->qpos + handQPosAdr, MANO_DOFS);
        mju_copy(data->qpos + handQPosAdr, splineQPos.data(), MANO_DOFS);
        mj_kinematics(model, data);
        mju_copy(data->mocap_pos + 3, data->xpos + handWristXPosOffset,
                 3 * (model->nmocap - 1));
        mju_copy(data->mocap_quat + 4, data->xquat + handWristXQuatOffset,
                 4 * (model->nmocap - 1));
        mju_copy(data->qpos + handQPosAdr, m_hand_kinematic_buffer, MANO_DOFS);
        mj_kinematics(model, data);

        // Contact loading
        mju_zero(model->site_pos, MAX_CONTACTS * XYZ_BLOCK_SIZE * 2);
        mju_zero(m_residual.m_r_hand_contact_position_buffer,
                 MAX_CONTACTS * XYZ_BLOCK_SIZE);
        mju_zero(m_residual.m_r_object_contact_position_buffer,
                 MAX_CONTACTS * XYZ_BLOCK_SIZE);

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
            mjtNum *handContactBlock =
                model->numeric_data + model->numeric_adr[handContactDataIndex];

            int handBodyIndex = handContactBlock[0];
            double localCoords[3] = {handContactBlock[1], handContactBlock[2],
                                     handContactBlock[3]};

            int siteRelativeOffset =
                handContactDataIndex - handContactDataStart;
            int fullSiteOffset =
                (handContactStartSiteId + siteRelativeOffset) * 3;

            mj_local2Global(data, model->site_pos + fullSiteOffset, nullptr,
                            localCoords, nullptr, handBodyIndex, 0);
        }

        mj_kinematics(model, data);

        m_residual.m_num_active_contacts = numActiveContacts;

        mju_copy(m_residual.m_r_hand_contact_position_buffer, data->site_xpos,
                 MAX_CONTACTS * XYZ_BLOCK_SIZE);
        mju_copy(m_residual.m_r_object_contact_position_buffer,
                 data->site_xpos + MAX_CONTACTS * XYZ_BLOCK_SIZE,
                 MAX_CONTACTS * XYZ_BLOCK_SIZE);

        double loopedQueryTime = fmod(data->time, m_spline_loopback_time);

        // Reset
        if (loopedQueryTime == 0)
        {
            int simObjBodyId =
                mj_name2id(model, mjOBJ_BODY, m_object_sim_body_name.c_str());
            int simObjDofs = model->nq - MANO_DOFS;

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
                    mju_copy4(data->qpos + objQposadr + 3,
                              splineObjectPos.data() + 3);
                }
                else
                {
                    // Otherwise zero out the configuration
                    mju_zero(data->qvel + objQposadr, simObjDofs);
                }
            }

            mju_copy(data->qpos + handQPosAdr, splineQPos.data(), MANO_DOFS);

            // Zero out entire system velocity, acceleration, and forces
            mju_zero(data->qvel, model->nv);
            mju_zero(data->qacc, model->nv);
            mju_zero(data->ctrl, model->nu);
            mju_zero(data->actuator_force, model->nu);
            mju_zero(data->qfrc_applied, model->nv);
            mju_zero(data->xfrc_applied, model->nbody * 6);
        }
    }

    MANOTask::MANOTask(string objectSimBodyName, string handTrajSplineFile,
                       string objectTrajSplineFile, string pcHandTrajSplineFile,
                       double startClampOffsetX, double startClampOffsetY,
                       double startClampOffsetZ, int totalFrames,
                       string objectContactStartDataName,
                       string handContactStartDataName)
        : m_residual(this), m_object_sim_body_name(objectSimBodyName),
          m_total_frames(totalFrames),
          m_object_contact_start_data_name(objectContactStartDataName),
          m_hand_contact_start_data_name(handContactStartDataName)
    {
        m_doftype_property_mappings["rotation"] = DofType::DOF_TYPE_ROTATION;
        m_doftype_property_mappings["rotationBallX"] =
            DofType::DOF_TYPE_ROTATION_BALL_X;
        m_doftype_property_mappings["rotationBallY"] =
            DofType::DOF_TYPE_ROTATION_BALL_Y;
        m_doftype_property_mappings["rotationBallZ"] =
            DofType::DOF_TYPE_ROTATION_BALL_Z;
        m_doftype_property_mappings["translation"] =
            DofType::DOF_TYPE_TRANSLATION;

        m_measurement_units_property_mappings["radians"] =
            MeasurementUnits::ROT_UNIT_RADIANS;
        m_measurement_units_property_mappings["degrees"] =
            MeasurementUnits::ROT_UNIT_DEGREES;
        m_measurement_units_property_mappings["meters"] =
            MeasurementUnits::TRANS_UNIT_METERS;
        m_measurement_units_property_mappings["centimeters"] =
            MeasurementUnits::TRANS_UNIT_CENTIMETERS;
        m_measurement_units_property_mappings["millimeters"] =
            MeasurementUnits::TRANS_UNIT_MILLIMETERS;

        m_start_clamp_offset[0] = startClampOffsetX;
        m_start_clamp_offset[1] = startClampOffsetY;
        m_start_clamp_offset[2] = startClampOffsetZ;

        Document dFullHandSplines = loadJSON(handTrajSplineFile);
        Document dObjectSplines = loadJSON(objectTrajSplineFile);
        Document dPcSplines = loadJSON(pcHandTrajSplineFile);

        m_spline_dimension = dFullHandSplines["dimension"].GetInt();
        m_spline_degree = dFullHandSplines["degree"].GetInt();
        m_spline_loopback_time = dFullHandSplines["time"].GetDouble();

        m_spline_loopback_time *= SLOWDOWN_FACTOR;

        m_num_pcs = dPcSplines["numComponents"].GetInt();

        for (const auto &splineData : dFullHandSplines["data"].GetArray())
        {
            int numControlPoints = splineData["numControlPoints"].GetInt();
            string dofType = splineData["type"].GetString();
            string units = splineData["units"].GetString();

            TrajectorySplineProperties properties;
            properties.numControlPoints = numControlPoints;
            properties.dofType = m_doftype_property_mappings[dofType];
            properties.units = m_measurement_units_property_mappings[units];

            vector<double> controlPoints;

            for (const auto &controlPointData :
                 splineData["controlPointData"].GetArray())
            {
                double entry = controlPointData.GetDouble();
                controlPoints.push_back(entry);
            }

            for (int i = 0; i < numControlPoints; i++)
            {
                controlPoints[i * 2] *= SLOWDOWN_FACTOR;
            }

            BSplineCurve<double> *bspc = new BSplineCurve<double>(
                m_spline_dimension, m_spline_degree, numControlPoints,
                m_doftype_property_mappings[dofType],
                m_measurement_units_property_mappings[units]);

            bspc->SetControlData(controlPoints);

            m_hand_traj_bspline_properties.push_back(properties);
            m_hand_traj_bspline_curves.push_back(bspc);
        }

        for (const auto &splineData : dObjectSplines["data"].GetArray())
        {
            int numControlPoints = splineData["numControlPoints"].GetInt();
            string dofType = splineData["type"].GetString();
            string units = splineData["units"].GetString();

            TrajectorySplineProperties properties;
            properties.numControlPoints = numControlPoints;
            properties.dofType = m_doftype_property_mappings[dofType];
            properties.units = m_measurement_units_property_mappings[units];

            vector<double> controlPoints;

            for (const auto &controlPointData :
                 splineData["controlPointData"].GetArray())
            {
                double entry = controlPointData.GetDouble();
                controlPoints.push_back(entry);
            }

            for (int i = 0; i < numControlPoints; i++)
            {
                controlPoints[i * 2] *= SLOWDOWN_FACTOR;
            }

            BSplineCurve<double> *bspc = new BSplineCurve<double>(
                m_spline_dimension, m_spline_degree, numControlPoints,
                m_doftype_property_mappings[dofType],
                m_measurement_units_property_mappings[units]);

            bspc->SetControlData(controlPoints);

            m_object_traj_bspline_properties.push_back(properties);
            m_object_traj_bspline_curves.push_back(bspc);
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

            TrajectorySplineProperties properties;
            properties.numControlPoints = numControlPoints;
            properties.dofType = m_doftype_property_mappings[dofType];
            properties.units = m_measurement_units_property_mappings[units];

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
                controlPoints[i * 2] *= SLOWDOWN_FACTOR;
            }

            BSplineCurve<double> *bspc = new BSplineCurve<double>(
                m_spline_dimension, m_spline_degree, numControlPoints,
                m_doftype_property_mappings[dofType],
                m_measurement_units_property_mappings[units]);

            bspc->SetControlData(controlPoints);

            m_hand_pc_traj_bspline_properties.push_back(properties);
            m_hand_pc_traj_bspline_curves.push_back(bspc);
        }

        mju_transpose(m_hand_pc_component_matrix.data(),
                      m_hand_pc_component_matrix.data(), m_num_pcs,
                      MANO_NON_ROOT_VEL_DOFS);

        if (m_hand_traj_bspline_curves.size() != MANO_VEL_DOFS)
        {
            cout << "ERROR: Expected " << MANO_VEL_DOFS << " dofs but read "
                 << m_hand_traj_bspline_curves.size() << " dofs." << endl;
        }
        else
        {
            cout << "Done" << endl;
        }
    }

    vector<double> MANOTask::GetDesiredAgentState(double time) const
    {
        vector<double> desiredState;

        double queryTime = fmod(time, m_spline_loopback_time);
        double parametricTime = queryTime / m_spline_loopback_time;

        double eulerAngles[3] = {0.0, 0.0, 0.0};
        double curveValue[2];

        for (int i = 0; i < 6; i++)
        {
            TrajectorySplineProperties properties =
                m_hand_traj_bspline_properties[i];

            m_hand_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            double dofValue = curveValue[1];

            switch (properties.dofType)
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

        // Correct for start clamp offset
        mju_sub3(desiredState.data(), desiredState.data(),
                 m_start_clamp_offset);

        // Convert root rotation to quaternion
        double quat[4];
        ConvertEulerAnglesToQuat(eulerAngles, quat);

        for (int i = 0; i < 4; i++)
        {
            desiredState.push_back(quat[i]);
        }

        // All remaining dofs are ball joints
        for (int i = 6; i < MANO_VEL_DOFS; i += 3)
        {
            mju_zero3(eulerAngles);
            mju_zero4(quat);

            for (int j = i; j < i + 3; j++)
            {
                TrajectorySplineProperties properties =
                    m_hand_traj_bspline_properties[j];

                m_hand_traj_bspline_curves[j]->GetPositionInMeasurementUnits(
                    parametricTime, curveValue);

                double dofValue = curveValue[1];

                switch (properties.dofType)
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
                    cout << "ERROR: Unknown non-root dof type" << endl;
                    break;
                }
            }

            ConvertEulerAnglesToQuat(eulerAngles, quat);

            for (int j = 0; j < 4; j++)
            {
                desiredState.push_back(quat[j]);
            }
        }

        return desiredState;
    }

    vector<double> MANOTask::GetDesiredAgentStateFromPCs(double time) const
    {
        vector<double> desiredState;

        double queryTime = fmod(time, m_spline_loopback_time);
        double parametricTime = queryTime / m_spline_loopback_time;

        // Get root state from original spline - no PC is performed on it
        double eulerAngles[3] = {0.0, 0.0, 0.0};
        double curveValue[2];

        for (int i = 0; i < 6; i++)
        {
            TrajectorySplineProperties properties =
                m_hand_traj_bspline_properties[i];

            m_hand_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            double dofValue = curveValue[1];

            switch (properties.dofType)
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

        // Correct for start clamp offset
        mju_sub3(desiredState.data(), desiredState.data(),
                 m_start_clamp_offset);

        // Convert root rotation to quaternion
        double quat[4];
        ConvertEulerAnglesToQuat(eulerAngles, quat);

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
        uncompressedState.resize(MANO_NON_ROOT_VEL_DOFS);

        mju_mulMatVec(uncompressedState.data(),
                      m_hand_pc_component_matrix.data(), pcState.data(),
                      MANO_NON_ROOT_VEL_DOFS, m_num_pcs);

        mju_addTo(uncompressedState.data(), m_hand_pc_center.data(),
                  MANO_NON_ROOT_VEL_DOFS);

        // All remaining dofs are ball joints
        for (int i = 0; i < MANO_NON_ROOT_VEL_DOFS; i += 3)
        {
            mju_zero3(eulerAngles);
            mju_zero4(quat);

            for (int j = i; j < i + 3; j++)
            {
                TrajectorySplineProperties properties =
                    m_hand_traj_bspline_properties[6 + j];

                double dofValue = uncompressedState[j];

                switch (properties.dofType)
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
                    cout << "ERROR: Unknown non-root dof type" << endl;
                    break;
                }
            }

            ConvertEulerAnglesToQuat(eulerAngles, quat);

            for (int j = 0; j < 4; j++)
            {
                desiredState.push_back(quat[j]);
            }
        }

        return desiredState;
    }

    vector<double> MANOTask::GetDesiredObjectState(double time) const
    {
        vector<double> desiredState;

        double queryTime = fmod(time, m_spline_loopback_time);
        double parametricTime = queryTime / m_spline_loopback_time;

        double eulerAngles[3] = {0.0, 0.0, 0.0};
        double curveValue[2];

        // Mopcap single body rigid object will have exactly 6 dofs
        for (int i = 0; i < 6; i++)
        {
            TrajectorySplineProperties properties =
                m_hand_traj_bspline_properties[i];

            m_object_traj_bspline_curves[i]->GetPositionInMeasurementUnits(
                parametricTime, curveValue);

            double dofValue = curveValue[1];

            switch (properties.dofType)
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
        double quat[4];
        ConvertEulerAnglesToQuat(eulerAngles, quat);

        for (int i = 0; i < 4; i++)
        {
            desiredState.push_back(quat[i]);
        }

        return desiredState;
    }

    vector<vector<double>> MANOTask::GetAgentBSplineControlData(
        int &dimension, int &degree, double &loopbackTime,
        double translationOffset[3], vector<DofType> &dofTypes,
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

            TrajectorySplineProperties properties =
                m_hand_traj_bspline_properties[i];

            int numElements = dataOriginal.size();

            // Intentionally avoid memcpy
            for (int j = 0; j < numElements; j++)
            {
                dataCopy.push_back(dataOriginal[j]);
            }

            bsplineControlData.push_back(dataCopy);
            dofTypes.push_back(properties.dofType);
            measurementUnits.push_back(properties.units);
        }

        return bsplineControlData;
    }

    vector<vector<double>> MANOTask::GetAgentPCBSplineControlData(
        int &dimension, int &degree, double &loopbackTime, int &numMaxPCs,
        vector<double> &centerData, vector<double> &componentData,
        double translationOffset[3], vector<DofType> &dofTypes,
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

            TrajectorySplineProperties properties =
                m_hand_traj_bspline_properties[i];

            int numElements = dataOriginal.size();

            // Intentionally avoid memcpy
            for (int j = 0; j < numElements; j++)
            {
                dataCopy.push_back(dataOriginal[j]);
            }

            bsplineControlData.push_back(dataCopy);
            dofTypes.push_back(properties.dofType);
            measurementUnits.push_back(properties.units);
        }

        for (int i = 0; i < m_num_pcs; i++)
        {
            vector<double> dataCopy; // Deep copy to avoid modifying original

            vector<double> dataOriginal =
                m_hand_pc_traj_bspline_curves[i]->GetControlData();

            TrajectorySplineProperties properties =
                m_hand_pc_traj_bspline_properties[i];

            int numElements = dataOriginal.size();

            // Intentionally avoid memcpy
            for (int j = 0; j < numElements; j++)
            {
                dataCopy.push_back(dataOriginal[j]);
            }

            bsplineControlData.push_back(dataCopy);
            dofTypes.push_back(properties.dofType);
            measurementUnits.push_back(properties.units);
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

    string MANOAppleTask::XmlPath() const
    {
        return GetModelPath("MANO/task_apple.xml");
    }

    string MANOAppleTask::Name() const { return "MANO Apple Pass"; }

    string MANODoorknobTask::XmlPath() const
    {
        return GetModelPath("MANO/task_doorknob.xml");
    }

    string MANODoorknobTask::Name() const { return "MANO Doorknob Use"; }

} // namespace mjpc
