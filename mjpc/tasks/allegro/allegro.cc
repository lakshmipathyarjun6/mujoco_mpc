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
    //     Residual (2): hand_state - hand_mocap_state
    // ------------------------------------------------------------

    // NOTE: Currently unclear how to adapt to non-free objects (e.g. doorknob)
    void AllegroTask::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                           double *residual) const
    {
        int offset = 0;

        // ---------- Residual (0) ----------
        // goal position
        double *goal_position = SensorByName(model, data, "object_traj_position");

        // system's position
        double *position = SensorByName(model, data, "object_position");

        // position error
        mju_sub3(residual + offset, position, goal_position);
        offset += 3;

        // ---------- Residual (1) ----------
        // goal orientation
        double *goal_orientation = SensorByName(model, data, "object_traj_orientation");

        // system's orientation
        double *orientation = SensorByName(model, data, "object_orientation");

        mju_normalize4(goal_orientation);
        mju_normalize4(orientation);

        // orientation error
        mju_subQuat(residual + offset, goal_orientation, orientation);
        offset += 3;

        // ---------- Residual (2) ----------
        int handRootBody = mj_name2id(model, mjOBJ_BODY, "wrist");
        int handQposadr = model->jnt_qposadr[model->body_jntadr[handRootBody]];

        mju_sub(residual + offset, data->qpos + handQposadr, r_qpos_buffer_, ALLEGRO_DOFS);

        offset += ALLEGRO_DOFS;

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

        string handKeyframeName = task_frame_prefix_ + "_hand_" + to_string(current_index + 1);
        string objectKeyframeName = task_frame_prefix_ + "_object_" + to_string(current_index + 1);

        double *objectKeyframeMPos = KeyMPosByName(model, data, objectKeyframeName);
        double *objectKeyframeMQuat = KeyMQuatByName(model, data, objectKeyframeName);

        double *handKeyframeQPos = KeyQPosByName(model, data, handKeyframeName);

        mju_copy(residual_.r_qpos_buffer_, handKeyframeQPos, ALLEGRO_DOFS);

        // // DEBUGGING: Uncomment below to view kinematic trajectory
        // int handRootBody = mj_name2id(model, mjOBJ_BODY, "wrist");
        // int handQposadr = model->jnt_qposadr[model->body_jntadr[handRootBody]];
        // mju_copy(data->qpos + handQposadr, handKeyframeQPos, ALLEGRO_DOFS);

        // Reset
        if (current_index == 0)
        {
            int objBody = mj_name2id(model, mjOBJ_BODY, sim_body_name_.c_str());
            int objDofs = model->nq - ALLEGRO_DOFS;
            bool objectSimBodyExists = objBody != -1;

            if (objectSimBodyExists)
            {
                int objQposadr = model->jnt_qposadr[model->body_jntadr[objBody]];

                // Free joint is special since the system can't be "zeroed out"
                // due to it needing to be based off the world frame
                if (objDofs == 7)
                {
                    // Reset configuration to first mocap frame
                    mju_copy3(data->qpos + objQposadr, objectKeyframeMPos);
                    mju_copy4(data->qpos + objQposadr + 3, objectKeyframeMQuat);
                }
                else
                {
                    // Otherwise zero out the configuration
                    mju_zero(data->qvel + objQposadr, objDofs);
                }
            }

            int handRootBody = mj_name2id(model, mjOBJ_BODY, "wrist");
            int handQposadr = model->jnt_qposadr[model->body_jntadr[handRootBody]];
            mju_copy(data->qpos + handQposadr, handKeyframeQPos, ALLEGRO_DOFS);

            // Zero out entire system velocity
            mju_zero(data->qvel, model->nq);
        }

        mju_copy3(data->mocap_pos, objectKeyframeMPos);
        mju_copy4(data->mocap_quat, objectKeyframeMQuat);
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
