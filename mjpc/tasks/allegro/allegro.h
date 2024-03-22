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

#include "JSONUtils.hpp"

#define FPS 12
#define SLOWDOWN_FACTOR 10

#define ABSOLUTE_MAX_CONTACT_SITES 6500
#define ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE ABSOLUTE_MAX_CONTACT_SITES * 3
#define ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE ABSOLUTE_MAX_CONTACT_SITES * 3 * 2

// Not equal due to root quaternion
#define ALLEGRO_DOFS 23
#define ALLEGRO_VEL_DOFS 22
#define ALLEGRO_NON_ROOT_DOFS 16

#define ALLEGRO_ROOT "wrist"
#define ALLEGRO_MOCAP_ROOT "palm"

#define OBJECT_CURRENT_POSITION "object_position"
#define OBJECT_CURRENT_ORIENTATION "object_orientation"

#define CONTACT_SITE_DATA_COUNT_FIELD_SIZE                                     \
    2 // first number gives the offset index, second number gives the number of
      // discrete points

#define SITE_DATA_START_NAME "contact_numdata_0"
#define OBJECT_CONTACT_START_SITE_NAME "contact_site_object_0"
#define HAND_CONTACT_START_SITE_NAME "contact_site_hand_0"

using namespace std;

namespace mjpc
{
    struct TrajectorySplineProperties
    {
        int numControlPoints;
        DofType dofType;
        MeasurementUnits units;
    };

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
                fill(begin(m_r_object_mocap_pos_buffer),
                     end(m_r_object_mocap_pos_buffer), 0);
                fill(begin(m_r_object_mocap_quat_buffer),
                     end(m_r_object_mocap_quat_buffer), 0);
                fill(begin(m_r_contact_indicator_buffer),
                     end(m_r_contact_indicator_buffer), 0);
                fill(begin(m_r_contact_position_buffer),
                     end(m_r_contact_position_buffer), 0);
            }

            explicit ResidualFn(
                const AllegroTask *task, const double qpos_buffer[ALLEGRO_DOFS],
                const double mocap_object_pos_buffer[3],
                const double mocap_object_quat_buffer[4],
                const double contact_indicator_buffer
                    [ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE],
                const double
                    contact_position_buffer[ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE])
                : mjpc::BaseResidualFn(task)
            {
                mju_copy(m_r_qpos_buffer, qpos_buffer, ALLEGRO_DOFS);
                mju_copy3(m_r_object_mocap_pos_buffer, mocap_object_pos_buffer);
                mju_copy4(m_r_object_mocap_quat_buffer,
                          mocap_object_quat_buffer);
                mju_copy(m_r_contact_indicator_buffer, contact_indicator_buffer,
                         ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE);
                mju_copy(m_r_contact_position_buffer, contact_position_buffer,
                         ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE);
            }

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            friend class AllegroTask;

            double m_r_qpos_buffer[ALLEGRO_DOFS];

            double m_r_object_mocap_pos_buffer[3];
            double m_r_object_mocap_quat_buffer[4];

            double m_r_contact_indicator_buffer
                [ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE];
            double
                m_r_contact_position_buffer[ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE];
        };

        AllegroTask(string objectSimBodyName, string handTrajSplineFile,
                    string pcHandTrajSplineFile, double startClampOffsetX,
                    double startClampOffsetY, double startClampOffsetZ,
                    int maxContactSites, string objectContactStartDataName,
                    string handContactStartDataName);

        vector<double> GetDesiredState(double time) const override;

        vector<double> GetDesiredStateFromPCs(double time) const override;

        vector<vector<double>> GetBSplineControlData(
            int &dimension, int &degree, double &loopbackTime,
            double translationOffset[3], vector<DofType> &dofTypes,
            vector<MeasurementUnits> &measurementUnits) const override;

        // --------------------- Transition for allegro task
        // ------------------------
        //   Set `data->mocap_pos` based on `data->time` to move the object
        //   site.
        // ---------------------------------------------------------------------------
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        unique_ptr<mjpc::ResidualFn> ResidualLocked() const override
        {
            return make_unique<ResidualFn>(
                this, m_residual.m_r_qpos_buffer,
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

        int m_spline_dimension;
        int m_spline_degree;

        vector<BSplineCurve<double> *> m_hand_traj_bspline_curves;
        vector<TrajectorySplineProperties> m_hand_traj_bspline_properties;

        int m_num_pcs;
        vector<double> m_hand_pc_center;
        vector<double> m_hand_pc_component_matrix;
        vector<BSplineCurve<double> *> m_hand_pc_traj_bspline_curves;
        vector<TrajectorySplineProperties> m_hand_pc_traj_bspline_properties;

        double m_spline_loopback_time;
        double m_start_clamp_offset[3];

        map<string, DofType> m_doftype_property_mappings;
        map<string, MeasurementUnits> m_measurement_units_property_mappings;
    };

    class AllegroAppleTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroAppleTask()
            : AllegroTask("apple_sim",
                          "/Users/arjunl/mujoco_mpc/mjpc/tasks/allegro/"
                          "splinetrajectories/apple_pass_1_hand.smexp",
                          "/Users/arjunl/mujoco_mpc/mjpc/tasks/allegro/"
                          "pcsplines/apple_pass_1.pcmexp",
                          -0.559216021990488, 1.0061246071752599,
                          1.3645857582385554, 1987,
                          "contact_pos_object_data_215_0",
                          "contact_pos_hand_data_215_0")
        {
        }

    private:
    };

    class AllegroDoorknobTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroDoorknobTask()
            : AllegroTask("doorknob_sim",
                          "/Users/arjunl/mujoco_mpc/mjpc/tasks/allegro/"
                          "splinetrajectories/doorknob_use_1_hand.smexp",
                          "/Users/arjunl/mujoco_mpc/mjpc/tasks/allegro/"
                          "pcsplines/doorknob_use_1.pcmexp",
                          -1.05350866, 0.30617798, 1.28931948, 6455,
                          "contact_pos_object_data_252_0",
                          "contact_pos_hand_data_252_0")
        {
        }
    };
} // namespace mjpc

#endif // MJPC_TASKS_ALLEGRO_HAND_H_
