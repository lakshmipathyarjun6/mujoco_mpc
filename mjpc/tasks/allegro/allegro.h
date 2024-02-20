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

#define FPS 12

#define ALLEGRO_DOFS 22
#define ALLEGRO_ROOT "wrist"
#define ALLEGRO_MOCAP_ROOT "palm"

#define OBJECT_TARGET_POSITION "object_traj_position"
#define OBJECT_TARGET_ORIENTATION "object_traj_orientation"

#define OBJECT_CURRENT_POSITION "object_position"
#define OBJECT_CURRENT_ORIENTATION "object_orientation"

#define CONTACT_SITE_DATA_COUNT_FIELD_SIZE 2 // first number gives the offset index, second number gives the number of discrete points

#define OBJECT_CONTACT_START_SITE_NAME "contact_site_object_0"
#define SITE_DATA_START_NAME "contact_numdata_0"

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
                fill(begin(m_r_qpos_buffer), end(m_r_qpos_buffer), 0);
            }

            explicit ResidualFn(const AllegroTask *task, const double qpos_buffer[ALLEGRO_DOFS])
                : mjpc::BaseResidualFn(task)
            {
                mju_copy(m_r_qpos_buffer, qpos_buffer, ALLEGRO_DOFS);
            }

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            friend class AllegroTask;

            double m_r_qpos_buffer[ALLEGRO_DOFS];
        };

        AllegroTask(string objectSimBodyName, int maxObjectContactSites, string objectContactStartDataName)
            : residual_(this), m_object_sim_body_name(objectSimBodyName),
              m_max_object_contact_sites(maxObjectContactSites),
              m_object_contact_start_data_name(objectContactStartDataName) {}

        // --------------------- Transition for allegro task ------------------------
        //   Set `data->mocap_pos` based on `data->time` to move the object site.
        // ---------------------------------------------------------------------------
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        unique_ptr<mjpc::ResidualFn> ResidualLocked() const override
        {
            return make_unique<ResidualFn>(this, residual_.m_r_qpos_buffer);
        }
        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;

        string m_object_sim_body_name;

        int m_max_object_contact_sites;
        string m_object_contact_start_data_name;

        double m_hand_kinematic_buffer[ALLEGRO_DOFS];
    };

    class AllegroAppleTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroAppleTask() : AllegroTask("apple_sim", 1987, "contact_pos_object_data_215_0") {}

    private:
    };

    class AllegroDoorknobTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroDoorknobTask() : AllegroTask("doorknob_sim", 6455, "contact_pos_object_data_252_0") {}
    };
} // namespace mjpc

#endif // MJPC_TASKS_ALLEGRO_HAND_H_
