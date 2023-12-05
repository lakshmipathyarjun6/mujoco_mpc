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

using namespace std;

namespace mjpc
{
    class Allegro : public Task
    {
    public:
        std::string Name() const override;
        std::string XmlPath() const override;

        class ResidualFn : public mjpc::BaseResidualFn
        {
        public:
            explicit ResidualFn(const Allegro *task, int current_mode = 0,
                                double reference_time = 0)
                : mjpc::BaseResidualFn(task),
                  current_mode_(current_mode),
                  reference_time_(reference_time) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            friend class Allegro;
            int current_mode_;
            double reference_time_;
        };

        Allegro() : residual_(this) {}

        // --------------------- Transition for allegro task ------------------------
        //   Set `data->mocap_pos` based on `data->time` to move the object site.
        // ---------------------------------------------------------------------------
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override
        {
            return std::make_unique<ResidualFn>(this, residual_.current_mode_,
                                                residual_.reference_time_);
        }
        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };
} // namespace mjpc

#endif // MJPC_TASKS_ALLEGRO_HAND_H_
