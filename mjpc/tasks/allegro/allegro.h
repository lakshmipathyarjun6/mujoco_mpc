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

#define ALLEGRO_DEFAULT_MOCAP_FPS 120
#define ALLEGRO_SLOWDOWN_FACTOR 10

#define ABSOLUTE_MAX_CONTACT_SITES 6500
#define ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE ABSOLUTE_MAX_CONTACT_SITES * 3
#define ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE ABSOLUTE_MAX_CONTACT_SITES * 3 * 2

// Not equal due to root quaternion
#define ALLEGRO_DOFS 23
#define ALLEGRO_VEL_DOFS 22
#define ALLEGRO_NON_ROOT_VEL_DOFS 16

#define XYZ_BLOCK_SIZE 3
#define QUAT_BLOCK_SIZE 4

#define ALLEGRO_ROOT "wrist"
#define ALLEGRO_MOCAP_ROOT "palm"

#define OBJECT_CURRENT_POSITION "object_position"
#define OBJECT_CURRENT_ORIENTATION "object_orientation"

#define SITE_DATA_START_NAME "contact_numdata_0"
#define OBJECT_CONTACT_START_SITE_NAME "contact_site_object_0"
#define HAND_CONTACT_START_SITE_NAME "contact_site_hand_0"

#define ALLEGRO_MAX_CONTACTS 21

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
                : mjpc::BaseResidualFn(task), m_bspline_loopback_time(0.0)
            {
                m_hand_traj_bspline_curves.clear();
                m_hand_traj_bspline_properties.clear();

                m_object_traj_bspline_curves.clear();
                m_object_traj_bspline_properties.clear();

                fill(begin(m_start_clamp_offset), end(m_start_clamp_offset), 0);

                fill(begin(m_r_contact_indicator_buffer),
                     end(m_r_contact_indicator_buffer), 0);
                fill(begin(m_r_contact_position_buffer),
                     end(m_r_contact_position_buffer), 0);
            }

            explicit ResidualFn(
                const AllegroTask *task, double bspline_loopback_time,
                const double start_clamp_offset[XYZ_BLOCK_SIZE],
                const vector<BSplineCurve<double> *> hand_traj_bspline_curves,
                const vector<TrajectorySplineProperties *>
                    hand_traj_bspline_properties,
                const vector<BSplineCurve<double> *> object_traj_bspline_curves,
                const vector<TrajectorySplineProperties *>
                    object_traj_bspline_properties,
                const double contact_indicator_buffer
                    [ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE],
                const double
                    contact_position_buffer[ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE])
                : mjpc::BaseResidualFn(task)
            {
                m_bspline_loopback_time = bspline_loopback_time;

                m_hand_traj_bspline_curves.clear();
                m_hand_traj_bspline_properties.clear();

                m_object_traj_bspline_curves.clear();
                m_object_traj_bspline_properties.clear();

                int numHandCurves = hand_traj_bspline_curves.size();

                for (int i = 0; i < numHandCurves; i++)
                {
                    m_hand_traj_bspline_curves.push_back(
                        hand_traj_bspline_curves[i]);

                    m_hand_traj_bspline_properties.push_back(
                        hand_traj_bspline_properties[i]);
                }

                int numObjectCurves = object_traj_bspline_curves.size();

                for (int i = 0; i < numObjectCurves; i++)
                {
                    m_object_traj_bspline_curves.push_back(
                        object_traj_bspline_curves[i]);

                    m_object_traj_bspline_properties.push_back(
                        object_traj_bspline_properties[i]);
                }

                mju_copy3(m_start_clamp_offset, start_clamp_offset);

                mju_copy(m_r_contact_indicator_buffer, contact_indicator_buffer,
                         ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE);
                mju_copy(m_r_contact_position_buffer, contact_position_buffer,
                         ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE);
            }

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

            // The residual function must be able to perform its own bspline
            // state queries since it will be called upon by the trajectory to
            // evaluate a future state cost

            vector<double> GetDesiredAgentState(double time) const;

            vector<double> GetDesiredObjectState(double time) const;

        private:
            friend class AllegroTask;

            double m_bspline_loopback_time;
            double m_start_clamp_offset[XYZ_BLOCK_SIZE];

            vector<BSplineCurve<double> *> m_hand_traj_bspline_curves;
            vector<TrajectorySplineProperties *> m_hand_traj_bspline_properties;

            vector<BSplineCurve<double> *> m_object_traj_bspline_curves;
            vector<TrajectorySplineProperties *>
                m_object_traj_bspline_properties;

            double m_r_contact_indicator_buffer
                [ABSOLUTE_MAX_CONTACT_RESULT_BUFF_SIZE];
            double
                m_r_contact_position_buffer[ABSOLUTE_MAX_CONTACT_POS_BUFF_SIZE];
        };

        AllegroTask(string objectSimBodyName, string handTrajSplineFile,
                    string objectTrajSplineFile, string pcHandTrajSplineFile,
                    double startClampOffsetX, double startClampOffsetY,
                    double startClampOffsetZ, int totalFrames,
                    string objectContactStartDataName,
                    string handContactStartDataName);

        vector<double> GetDesiredAgentState(double time) const;

        vector<double> GetDesiredAgentStateFromPCs(double time) const;

        vector<double> GetDesiredObjectState(double time) const;

        vector<vector<double>> GetAgentBSplineControlData(
            int &dimension, int &degree, double &loopbackTime,
            double translationOffset[3], vector<DofType> &dofTypes,
            vector<MeasurementUnits> &measurementUnits) const override;

        vector<vector<double>> GetAgentPCBSplineControlData(
            int &dimension, int &degree, double &loopbackTime, int &numMaxPCs,
            vector<double> &centerData, vector<double> &componentData,
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
                this, m_residual.m_bspline_loopback_time,
                m_residual.m_start_clamp_offset,
                m_residual.m_hand_traj_bspline_curves,
                m_residual.m_hand_traj_bspline_properties,
                m_residual.m_object_traj_bspline_curves,
                m_residual.m_object_traj_bspline_properties,
                m_residual.m_r_contact_indicator_buffer,
                m_residual.m_r_contact_position_buffer);
        }
        ResidualFn *InternalResidual() override { return &m_residual; }

    private:
        ResidualFn m_residual;

        string m_object_sim_body_name;

        int m_total_frames;
        string m_object_contact_start_data_name;
        string m_hand_contact_start_data_name;

        int m_spline_dimension;
        int m_spline_degree;

        vector<BSplineCurve<double> *> m_hand_traj_bspline_curves;
        vector<TrajectorySplineProperties *> m_hand_traj_bspline_properties;

        vector<BSplineCurve<double> *> m_object_traj_bspline_curves;
        vector<TrajectorySplineProperties *> m_object_traj_bspline_properties;

        int m_num_pcs;
        vector<double> m_hand_pc_center;
        vector<double> m_hand_pc_component_matrix;
        vector<BSplineCurve<double> *> m_hand_pc_traj_bspline_curves;
        vector<TrajectorySplineProperties *> m_hand_pc_traj_bspline_properties;

        double m_spline_loopback_time;
        double m_start_clamp_offset[XYZ_BLOCK_SIZE];
    };

    class AllegroApplePassTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroApplePassTask()
            : AllegroTask(
                  "apple_sim",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/allegro/"
                  "splinetrajectories/apple_pass_1_hand.smexp",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/"
                  "shared_spline_trajectories/apple_pass_1_object.smexp",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/allegro/"
                  "pcsplines/apple_pass_1.pcmexp",
                  -0.559216021990488, 1.0061246071752599, 1.3645857582385554,
                  703, "contact_pos_object_data_215_0",
                  "contact_pos_hand_data_215_0")
        {
        }

    private:
    };

    class AllegroDoorknobUseTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroDoorknobUseTask()
            : AllegroTask(
                  "doorknob_sim",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/allegro/"
                  "splinetrajectories/doorknob_use_1_hand.smexp",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/"
                  "shared_spline_trajectories/doorknob_use_1_object.smexp",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/allegro/"
                  "pcsplines/doorknob_use_1.pcmexp",
                  -1.05350866, 0.30617798, 1.28931948, 1040,
                  "contact_pos_object_data_252_0",
                  "contact_pos_hand_data_252_0")
        {
        }
    };
} // namespace mjpc

#endif // MJPC_TASKS_ALLEGRO_HAND_H_
