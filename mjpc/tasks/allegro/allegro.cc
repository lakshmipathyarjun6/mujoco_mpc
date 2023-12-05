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
    703 // Apple Pass 1
};

const string kTaskPrefixes[] = {
    "apple_pass_1"};

// return length of motion trajectory
int MotionLength(int id) { return kMotionLengths[id]; }

string MotionPrefix(int id) { return kTaskPrefixes[id]; }

// return starting keyframe index for motion
int MotionStartIndex(int id)
{
    int start = 0;

    for (int i = 0; i < id; i++)
    {
        start += MotionLength(i);
    }

    return start;
}

namespace mjpc
{
    std::string Allegro::XmlPath() const
    {
        return GetModelPath("allegro/task.xml");
    }
    std::string Allegro::Name() const { return "Allegro"; }

    void Allegro::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {}

    // --------------------- Transition for allegro task ------------------------
    //   Set `data->mocap_pos` based on `data->time` to move the object site.
    // ---------------------------------------------------------------------------
    void Allegro::TransitionLocked(mjModel *model, mjData *data)
    {
        // get motion start index
        int start = MotionStartIndex(mode);

        // get motion trajectory length
        int length = MotionLength(mode);

        // get motion trajectory prefix
        string motionPrefix = MotionPrefix(mode);

        // check for motion switch
        if (residual_.current_mode_ != mode || data->time == 0.0)
        {
            residual_.current_mode_ = mode;         // set motion id
            residual_.reference_time_ = data->time; // set reference time
        }

        // indices
        double rounded_index = floor((data->time - residual_.reference_time_) * kFps);

        int current_index = (int(rounded_index) + start) % length;

        string keyframeName = motionPrefix + "_object_" + to_string(current_index + 1);

        double *keyframePos = KeyMPosByName(model, data, keyframeName);
        double *keyframeQuat = KeyMQuatByName(model, data, keyframeName);

        mju_copy3(data->mocap_pos, keyframePos);
        mju_copy4(data->mocap_quat, keyframeQuat);
    }

} // namespace mjpc
