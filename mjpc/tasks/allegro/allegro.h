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

#define ABSOLUTE_MAX_CONTACT_SITES 6500
#define ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE ABSOLUTE_MAX_CONTACT_SITES * 3
#define ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE ABSOLUTE_MAX_CONTACT_SITES * 3 * 2

// Not equal due to root quaternion
#define ALLEGRO_DOFS 23
#define ALLEGRO_VEL_DOFS 22

#define ALLEGRO_ROOT "wrist"
#define ALLEGRO_MOCAP_ROOT "palm"

#define OBJECT_CURRENT_POSITION "object_position"
#define OBJECT_CURRENT_ORIENTATION "object_orientation"

#define CONTACT_SITE_DATA_COUNT_FIELD_SIZE 2 // first number gives the offset index, second number gives the number of discrete points

#define SITE_DATA_START_NAME "contact_numdata_0"
#define OBJECT_CONTACT_START_SITE_NAME "contact_site_object_0"
#define HAND_CONTACT_START_SITE_NAME "contact_site_hand_0"

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
                fill(begin(m_r_object_mocap_pos_buffer), end(m_r_object_mocap_pos_buffer), 0);
                fill(begin(m_r_object_mocap_quat_buffer), end(m_r_object_mocap_quat_buffer), 0);
                fill(begin(m_r_contact_indicator_buffer), end(m_r_contact_indicator_buffer), 0);
                fill(begin(m_r_contact_position_buffer), end(m_r_contact_position_buffer), 0);
            }

            explicit ResidualFn(
                const AllegroTask *task,
                const double qpos_buffer[ALLEGRO_DOFS],
                const double mocap_object_pos_buffer[3],
                const double mocap_object_quat_buffer[4],
                const double contact_indicator_buffer[ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE],
                const double contact_position_buffer[ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE])
                : mjpc::BaseResidualFn(task)
            {
                mju_copy(m_r_qpos_buffer, qpos_buffer, ALLEGRO_DOFS);
                mju_copy3(m_r_object_mocap_pos_buffer, mocap_object_pos_buffer);
                mju_copy4(m_r_object_mocap_quat_buffer, mocap_object_quat_buffer);
                mju_copy(m_r_contact_indicator_buffer, contact_indicator_buffer, ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE);
                mju_copy(m_r_contact_position_buffer, contact_position_buffer, ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE);
            }

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            friend class AllegroTask;

            double m_r_qpos_buffer[ALLEGRO_DOFS];

            double m_r_object_mocap_pos_buffer[3];
            double m_r_object_mocap_quat_buffer[4];

            double m_r_contact_indicator_buffer[ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE];
            double m_r_contact_position_buffer[ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE];
        };

        AllegroTask(string objectSimBodyName, int maxContactSites, string objectContactStartDataName, string handContactStartDataName)
            : m_residual(this), m_object_sim_body_name(objectSimBodyName),
              m_max_contact_sites(maxContactSites),
              m_object_contact_start_data_name(objectContactStartDataName),
              m_hand_contact_start_data_name(handContactStartDataName) {}

        // --------------------- Transition for allegro task ------------------------
        //   Set `data->mocap_pos` based on `data->time` to move the object site.
        // ---------------------------------------------------------------------------
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        unique_ptr<mjpc::ResidualFn> ResidualLocked() const override
        {
            return make_unique<ResidualFn>(
                this,
                m_residual.m_r_qpos_buffer,
                m_residual.m_r_object_mocap_pos_buffer,
                m_residual.m_r_object_mocap_quat_buffer,
                m_residual.m_r_contact_indicator_buffer,
                m_residual.m_r_contact_position_buffer);
        }
        ResidualFn *InternalResidual() override { return &m_residual; }

    private:
        ResidualFn m_residual;

        string m_object_sim_body_name;

        int m_max_contact_sites;

        string m_object_contact_start_data_name;
        string m_hand_contact_start_data_name;

        double m_hand_kinematic_buffer[ALLEGRO_DOFS];
    };

    class AllegroAppleTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroAppleTask() : AllegroTask("apple_sim", 1987, "contact_pos_object_data_215_0", "contact_pos_hand_data_215_0") {}

    private:
    };

    class AllegroDoorknobTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroDoorknobTask() : AllegroTask("doorknob_sim", 6455, "contact_pos_object_data_252_0", "contact_pos_hand_data_252_0") {}
    };
} // namespace mjpc

#endif // MJPC_TASKS_ALLEGRO_HAND_H_
