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

#ifndef MJPC_TASKS_APPLE_H_
#define MJPC_TASKS_APPLE_H_

#include <mujoco/mujoco.h>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

#define APPLE_FPS 12

#define APPLE_DOFS 3

#define TARGET_POSITION "object_traj_position"
#define TARGET_ORIENTATION "object_traj_orientation"

#define CURRENT_POSITION "object_position"
#define CURRENT_ORIENTATION "object_orientation"

using namespace std;

namespace mjpc
{
    class AppleTask : public Task
    {
    public:
        class ResidualFn : public mjpc::BaseResidualFn
        {
        public:
            explicit ResidualFn(const AppleTask *task)
                : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            friend class AppleTask;
        };

        AppleTask() : residual_(this) {}

        // --------------------- Transition for allegro task ------------------------
        //   Set `data->mocap_pos` based on `data->time` to move the object site.
        // ---------------------------------------------------------------------------
        void TransitionLocked(mjModel *model, mjData *data) override;

        std::string Name() const override;
        std::string XmlPath() const override;

    protected:
        unique_ptr<mjpc::ResidualFn> ResidualLocked() const override
        {
            return make_unique<ResidualFn>(this);
        }
        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;

        int num_mocap_frames_ = 703;
    };
} // namespace mjpc

#endif // MJPC_TASKS_APPLE_H_
