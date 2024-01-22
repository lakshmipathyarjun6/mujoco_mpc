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

#include "mjpc/tasks/apple/apple.h"

namespace mjpc
{

    // ---------- Residuals for allegro hand manipulation task ---------
    //   Number of residuals: 4
    //     Residual (0): object_position - object_traj_position
    //     Residual (1): object_orientation - object_traj_orientation
    //     Residual (2): hand_state - hand_mocap_state
    // ------------------------------------------------------------

    // NOTE: Currently unclear how to adapt to non-free objects (e.g. doorknob)
    void AppleTask::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                         double *residual) const
    {
        int offset = 0;

        // ---------- Residual (0) ----------
        // goal position
        double *goal_position = SensorByName(model, data, TARGET_POSITION);

        // system's position
        double *position = SensorByName(model, data, CURRENT_POSITION);

        // position error
        mju_sub3(residual + offset, position, goal_position);
        offset += 3;

        // sensor dim sanity check
        CheckSensorDim(model, offset);
    }

    // --------------------- Transition for apple task ------------------------
    //   Set `data->mocap_pos` based on `data->time` to move the object site.
    // ---------------------------------------------------------------------------
    void AppleTask::TransitionLocked(mjModel *model, mjData *data)
    {
        // indices
        double rounded_index = floor(data->time * APPLE_FPS);
        int current_index = int(rounded_index) % num_mocap_frames_;

        int objectMocapPosOffset = 3 * model->nmocap * current_index;

        // Reset
        if (current_index == 0)
        {
            // Reset configuration to first mocap frame
            mju_copy3(data->qpos, model->key_mpos + objectMocapPosOffset);

            // Zero out entire system velocity
            mju_zero(data->qvel, model->nq);
        }

        mju_copy3(data->mocap_pos, model->key_mpos + objectMocapPosOffset);
    }

    string AppleTask::XmlPath() const
    {
        return GetModelPath("apple/task.xml");
    }

    string AppleTask::Name() const { return "Simple Apple Pass"; }

} // namespace mjpc
