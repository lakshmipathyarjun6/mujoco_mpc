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
    //     Residual (2): hand_root_positions - hand_mocap_root_position
    //     Residual (3): hand_root_orientation - hand_mocap_root_orientation
    //     Residual (4): hand_state - hand_mocap_state
    // ------------------------------------------------------------

    // NOTE: Currently unclear how to adapt to non-free objects (e.g. doorknob)
    void AllegroTask::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                           double *residual) const
    {
        int offset = 0;

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
        int handRootBodyId = mj_name2id(model, mjOBJ_BODY, ALLEGRO_ROOT);
        int bodyJointAdr = model->body_jntadr[handRootBodyId];
        int handQPosAdr = model->jnt_qposadr[bodyJointAdr];

        mju_sub(residual + offset, data->qpos + handQPosAdr, r_qpos_buffer_, 3);

        offset += 3;

        // ---------- Residual (3) ----------
        int rootOffset = 3;
        mju_sub(residual + offset, data->qpos + handQPosAdr + rootOffset, r_qpos_buffer_ + rootOffset, 3);

        offset += 3;

        // ---------- Residual (4) ----------
        rootOffset = 6;
        mju_sub(residual + offset, data->qpos + handQPosAdr + rootOffset, r_qpos_buffer_ + rootOffset, ALLEGRO_DOFS - 6);

        offset += ALLEGRO_DOFS - 6;

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
        int current_index = int(rounded_index) % num_mocap_frames_;

        // current_index = min(current_index, 1);

        int handMocapQOffset = model->nq * current_index;

        mju_copy(residual_.r_qpos_buffer_, model->key_qpos + handMocapQOffset, ALLEGRO_DOFS);

        int objectMocapPosOffset = 3 * model->nmocap * current_index;
        int objectMocapQuatOffset = 4 * model->nmocap * current_index;

        int handPalmBodyId = mj_name2id(model, mjOBJ_BODY, ALLEGRO_MOCAP_ROOT);
        int handPalmXPosOffset = 3 * handPalmBodyId;
        int handPalmXQuatOffset = 4 * handPalmBodyId;

        int handRootBodyId = mj_name2id(model, mjOBJ_BODY, ALLEGRO_ROOT);
        int bodyJointAdr = model->body_jntadr[handRootBodyId];
        int handQPosAdr = model->jnt_qposadr[bodyJointAdr];

        // DEBUG ONLY
        mju_copy(hand_kinematic_buffer_, data->qpos + handQPosAdr, ALLEGRO_DOFS);
        mju_copy(data->qpos + handQPosAdr, model->key_qpos + handMocapQOffset, ALLEGRO_DOFS);
        mj_kinematics(model, data);
        mju_copy(data->mocap_pos + 3, data->xpos + handPalmXPosOffset, 3 * (model->nmocap - 1));
        mju_copy(data->mocap_quat + 4, data->xquat + handPalmXQuatOffset, 4 * (model->nmocap - 1));
        mju_copy(data->qpos + handQPosAdr, hand_kinematic_buffer_, ALLEGRO_DOFS);
        mj_kinematics(model, data);

        // Reset
        if (current_index == 0)
        {
            int simObjBodyId = mj_name2id(model, mjOBJ_BODY, object_sim_body_name_.c_str());
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
