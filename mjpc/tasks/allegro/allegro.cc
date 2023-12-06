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

// Hardcoded constant matching keyframes from dataset.
constexpr double kFps = 120.0;

constexpr int kMotionLengths[] = {
    703, // Apple Pass 1
    1040 // Doorknob Use 1
};

const string kTaskPrefixes[] = {
    "apple_pass_1",  // Apple Pass 1
    "doorknob_use_1" // Doorknob Use 1
};

namespace mjpc
{
    AllegroTask::AllegroTask(int taskId) : residual_(this), task_id_(taskId)
    {
        num_mocap_frames_ = kMotionLengths[task_id_];
        task_frame_prefix_ = kTaskPrefixes[task_id_];
    }

    void AllegroTask::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                           double *residual) const {}

    // --------------------- Transition for allegro task ------------------------
    //   Set `data->mocap_pos` based on `data->time` to move the object site.
    // ---------------------------------------------------------------------------
    void AllegroTask::TransitionLocked(mjModel *model, mjData *data)
    {
        // indices
        double rounded_index = floor(data->time * kFps);
        int current_index = int(rounded_index) % num_mocap_frames_;

        string handKeyframeName = task_frame_prefix_ + "_hand_" + to_string(current_index + 1);
        string objectKeyframeName = task_frame_prefix_ + "_object_" + to_string(current_index + 1);

        double *objectKeyframeMPos = KeyMPosByName(model, data, objectKeyframeName);
        double *objectKeyframeMQuat = KeyMQuatByName(model, data, objectKeyframeName);

        double *handKeyframeQPos = KeyQPosByName(model, data, handKeyframeName);

        // DEBUG ONLY
        mju_copy(data->qpos, handKeyframeQPos, model->nq);

        // // Actual Reset
        // if (current_index == 0)
        // {
        //     double *handKeyframeQPos = KeyQPosByName(model, data, handKeyframeName);
        //     mju_copy(data->qpos, handKeyframeQPos, model->nu);
        // }

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
