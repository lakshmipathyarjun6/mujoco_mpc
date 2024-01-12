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

#ifndef MJPC_TASKS_ALLEGRO_HAND_H_
#define MJPC_TASKS_ALLEGRO_HAND_H_

#include <mujoco/mujoco.h>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

#define FPS 120
#define ALLEGRO_DOFS 23

using namespace std;

namespace mjpc
{
    class AllegroTask : public Task
    {
    public:
        class ResidualFn : public mjpc::BaseResidualFn
        {
        public:
            explicit ResidualFn(const AllegroTask *task)
                : mjpc::BaseResidualFn(task)
            {
                fill(begin(r_qpos_buffer_), end(r_qpos_buffer_), 0);
            }

            explicit ResidualFn(const AllegroTask *task, const double qpos_buffer[ALLEGRO_DOFS])
                : mjpc::BaseResidualFn(task)
            {
                mju_copy(r_qpos_buffer_, qpos_buffer, ALLEGRO_DOFS);
            }

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            friend class AllegroTask;

            double r_qpos_buffer_[ALLEGRO_DOFS];
        };

        AllegroTask(int numMocapFrames, string taskNamePrefix, string bodyName)
            : residual_(this), num_mocap_frames_(numMocapFrames),
              task_frame_prefix_(taskNamePrefix), sim_body_name_(bodyName) {}

        // --------------------- Transition for allegro task ------------------------
        //   Set `data->mocap_pos` based on `data->time` to move the object site.
        // ---------------------------------------------------------------------------
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        unique_ptr<mjpc::ResidualFn> ResidualLocked() const override
        {
            return make_unique<ResidualFn>(this, residual_.r_qpos_buffer_);
        }
        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;

        int num_mocap_frames_;
        string task_frame_prefix_;
        string sim_body_name_;
    };

    class AllegroAppleTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroAppleTask() : AllegroTask(703, "apple_pass_1", "apple_sim") {}
    };

    class AllegroDoorknobTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroDoorknobTask() : AllegroTask(1040, "doorknob_use_1", "doorknob_sim") {}
    };
} // namespace mjpc

#endif // MJPC_TASKS_ALLEGRO_HAND_H_
