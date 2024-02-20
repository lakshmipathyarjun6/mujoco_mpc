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
    void AllegroTask::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                           double *residual) const
    {
        int offset = 0;

        // bool agent_object_collision_detected = false;
        vector<mjContact *> agent_object_collisions;

        // Get all collisions
        for (int c = 0; c < data->ncon; c++)
        {
            int *colliding_geoms = data->contact[c].geom; // size 2

            string colliding_geom_1 = model->names + model->name_geomadr[colliding_geoms[0]];
            string colliding_geom_2 = model->names + model->name_geomadr[colliding_geoms[1]];

            // Check for agent-object collisions only
            // Breaking into separate clauses for readability
            if (
                colliding_geom_1.compare(0, AGENT_GEOM_COLLIDER_PREFIX.size(), AGENT_GEOM_COLLIDER_PREFIX) == 0 &&
                colliding_geom_2.compare(0, SIM_GEOM_COLLIDER_PREFIX.size(), SIM_GEOM_COLLIDER_PREFIX) == 0)
            {
                // agent_object_collision_detected = true;
                agent_object_collisions.push_back(&data->contact[c]);
            }
            else if (
                colliding_geom_2.compare(0, AGENT_GEOM_COLLIDER_PREFIX.size(), AGENT_GEOM_COLLIDER_PREFIX) == 0 &&
                colliding_geom_1.compare(0, SIM_GEOM_COLLIDER_PREFIX.size(), SIM_GEOM_COLLIDER_PREFIX) == 0)
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
        //         cout << "\tGeometry 1: " << model->names + model->name_geomadr[agent_object_collisions[c]->geom[0]] << endl;
        //         cout << "\tGeometry 2: " << model->names + model->name_geomadr[agent_object_collisions[c]->geom[1]] << endl;
        //     }
        // }

        // ---------- Residual (0) ----------
        // goal position
        double *goal_position = SensorByName(model, data, OBJECT_TARGET_POSITION);

        // system's position
        double *position = SensorByName(model, data, OBJECT_CURRENT_POSITION);

        // position error
        mju_sub3(residual + offset, position, goal_position);
        offset += 3;

        // ---------- Residual (1) ----------
        // goal orientation
        double *goal_orientation = SensorByName(model, data, OBJECT_TARGET_ORIENTATION);

        // system's orientation
        double *orientation = SensorByName(model, data, OBJECT_CURRENT_ORIENTATION);

        mju_normalize4(goal_orientation);
        mju_normalize4(orientation);

        // orientation error
        mju_subQuat(residual + offset, goal_orientation, orientation);
        offset += 3;

        // ---------- Residual (2) ----------
        mju_copy(residual + offset, data->qvel, ALLEGRO_DOFS);
        offset += ALLEGRO_DOFS;

        // // ---------- Residual (2) ----------
        // int handRootBodyId = mj_name2id(model, mjOBJ_BODY, ALLEGRO_ROOT);
        // int bodyJointAdr = model->body_jntadr[handRootBodyId];
        // int handQPosAdr = model->jnt_qposadr[bodyJointAdr];

        // mju_sub(residual + offset, data->qpos + handQPosAdr, m_r_qpos_buffer, 3);

        // offset += 3;

        // // ---------- Residual (3) ----------
        // int rootOffset = 3;
        // mju_sub(residual + offset, data->qpos + handQPosAdr + rootOffset, m_r_qpos_buffer + rootOffset, 3);

        // offset += 3;

        // // ---------- Residual (4) ----------
        // rootOffset = 6;
        // mju_sub(residual + offset, data->qpos + handQPosAdr + rootOffset, m_r_qpos_buffer + rootOffset, ALLEGRO_DOFS - 6);

        // offset += ALLEGRO_DOFS - 6;

        // sensor dim sanity check
        CheckSensorDim(model, offset);
    }

    // --------------------- Transition for allegro task ------------------------
    //   Set `data->mocap_pos` based on `data->time` to move the object site.
    // ---------------------------------------------------------------------------
    void AllegroTask::TransitionLocked(mjModel *model, mjData *data)
    {
        // indices
        double rounded_index = floor(data->time * FPS);
        mode = int(rounded_index) % model->nkey;

        // mode = min(mode, 1);

        int handMocapQOffset = model->nq * mode;

        mju_copy(residual_.m_r_qpos_buffer, model->key_qpos + handMocapQOffset, ALLEGRO_DOFS);

        int objectMocapPosOffset = 3 * model->nmocap * mode;
        int objectMocapQuatOffset = 4 * model->nmocap * mode;

        int handPalmBodyId = mj_name2id(model, mjOBJ_BODY, ALLEGRO_MOCAP_ROOT);
        int handPalmXPosOffset = 3 * handPalmBodyId;
        int handPalmXQuatOffset = 4 * handPalmBodyId;

        int handRootBodyId = mj_name2id(model, mjOBJ_BODY, ALLEGRO_ROOT);
        int bodyJointAdr = model->body_jntadr[handRootBodyId];
        int handQPosAdr = model->jnt_qposadr[bodyJointAdr];

        // DEBUG ONLY
        mju_copy(m_hand_kinematic_buffer, data->qpos + handQPosAdr, ALLEGRO_DOFS);
        mju_copy(data->qpos + handQPosAdr, model->key_qpos + handMocapQOffset, ALLEGRO_DOFS);
        mj_kinematics(model, data);
        mju_copy(data->mocap_pos + 3, data->xpos + handPalmXPosOffset, 3 * (model->nmocap - 1));
        mju_copy(data->mocap_quat + 4, data->xquat + handPalmXQuatOffset, 4 * (model->nmocap - 1));
        mju_copy(data->qpos + handQPosAdr, m_hand_kinematic_buffer, ALLEGRO_DOFS);
        mj_kinematics(model, data);

        int siteMetadataStartId = mj_name2id(model, mjOBJ_NUMERIC, SITE_DATA_START_NAME);
        int siteMetadataOffset = siteMetadataStartId + mode;

        mjtNum *metadataData = model->numeric_data + model->numeric_adr[siteMetadataOffset];
        int contactDataOffset = int(metadataData[0]);
        int contactDataSize = int(metadataData[1]);

        mju_zero(model->site_pos, m_max_contact_sites * 3 * 2);

        // Load object contact site data
        int objectContactStartSiteId = mj_name2id(model, mjOBJ_SITE, OBJECT_CONTACT_START_SITE_NAME);
        int objectContactDataStartId = mj_name2id(model, mjOBJ_NUMERIC, m_object_contact_start_data_name.c_str());

        int objectContactDataStart = objectContactDataStartId + contactDataOffset;

        mju_copy(model->site_pos + objectContactStartSiteId * 3, model->numeric_data + model->numeric_adr[objectContactDataStart], contactDataSize * 3);

        // Load hand contact site data
        // Doing this manually rather than running FK since reassembly and extra geoms is pointlessly expensive and convoluted
        int handContactStartSiteId = mj_name2id(model, mjOBJ_SITE, HAND_CONTACT_START_SITE_NAME);
        int handContactDataStartId = mj_name2id(model, mjOBJ_NUMERIC, m_hand_contact_start_data_name.c_str());

        int handContactDataStart = handContactDataStartId + contactDataOffset;

        for (int handContactDataIndex = handContactDataStart; handContactDataIndex < handContactDataStart + contactDataSize; handContactDataIndex++)
        {
            string handContactName = model->names + model->name_numericadr[handContactDataIndex];
            mjtNum *handContactBlock = model->numeric_data + model->numeric_adr[handContactDataIndex];

            int handBodyIndex = handContactBlock[0];
            double localCoords[3] = {handContactBlock[1], handContactBlock[2], handContactBlock[3]};

            int siteRelativeOffset = handContactDataIndex - handContactDataStart;
            int fullSiteOffset = (handContactStartSiteId + siteRelativeOffset) * 3;

            mj_local2Global(data, model->site_pos + fullSiteOffset,
                            nullptr, localCoords,
                            nullptr, handBodyIndex, 0);
        }
        mj_kinematics(model, data);

        // Reset
        if (mode == 0)
        {
            int simObjBodyId = mj_name2id(model, mjOBJ_BODY, m_object_sim_body_name.c_str());
            int simObjDofs = model->nq - ALLEGRO_DOFS;

            bool objectSimBodyExists = simObjBodyId != -1;

            if (objectSimBodyExists)
            {
                int objQposadr = model->jnt_qposadr[model->body_jntadr[simObjBodyId]];

                // Free joint is special since the system can't be "zeroed out"
                // due to it needing to be based off the world frame
                if (simObjDofs == 7)
                {
                    // Reset configuration to first mocap frame
                    mju_copy3(data->qpos + objQposadr, model->key_mpos + objectMocapPosOffset);
                    mju_copy4(data->qpos + objQposadr + 3, model->key_mquat + objectMocapQuatOffset);
                }
                else
                {
                    // Otherwise zero out the configuration
                    mju_zero(data->qvel + objQposadr, simObjDofs);
                }
            }

            mju_copy(data->qpos + handQPosAdr, model->key_qpos + handMocapQOffset, ALLEGRO_DOFS);

            // Zero out entire system velocity, acceleration, and forces
            mju_zero(data->qvel, model->nv);
            mju_zero(data->qacc, model->nv);
            mju_zero(data->ctrl, model->nu);
            mju_zero(data->actuator_force, model->nu);
            mju_zero(data->qfrc_applied, model->nv);
            mju_zero(data->xfrc_applied, model->nbody * 6);

            // Zero out all object contact sites
            mju_zero(model->site_pos, m_max_contact_sites * 3 * 2);

            for(int i = 0; i < model->nsite; i++)
            {
                model->site_sameframe[i] = 0;
            }
        }

        // Object mocap is first in config
        mju_copy3(data->mocap_pos, model->key_mpos + objectMocapPosOffset);
        mju_copy4(data->mocap_quat, model->key_mquat + objectMocapQuatOffset);
    }

    string AllegroAppleTask::XmlPath() const
    {
        return GetModelPath("allegro/task_apple.xml");
    }

    string AllegroAppleTask::Name() const { return "Allegro Apple Pass"; }

    string AllegroDoorknobTask::XmlPath() const
    {
        return GetModelPath("allegro/task_doorknob.xml");
    }

    string AllegroDoorknobTask::Name() const { return "Allegro Doorknob Use"; }

} // namespace mjpc
