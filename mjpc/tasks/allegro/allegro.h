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

// Apple near success: Slowdown factor 8
// Doorknob success: Slowdown factor 5-10
// Stapler success: Slowdown factor 7
// Waterbottle success: Slowdown factor 5

#define ALLEGRO_DEFAULT_MOCAP_FPS 120
#define ALLEGRO_DEFAULT_SLOWDOWN_FACTOR 10.0

// Not equal due to root quaternion
#define ALLEGRO_DOFS 23
#define ALLEGRO_VEL_DOFS 22
#define ALLEGRO_NON_ROOT_VEL_DOFS 16

#define XYZ_BLOCK_SIZE 3
#define QUAT_BLOCK_SIZE 4

#define ALLEGRO_ACTIVE_CONTACT_FAILURE_THRESHOLD 6
#define ALLEGRO_MAX_CONSECUTIVE_FAILURE_TOLERANCES 500

#define DATA_DUMP_FILE_NAME_PREFIX "agent_run_"
#define DATA_DUMP_FILE_TYPE ".json"

#define ALLEGRO_AGENT_NAME "allegro"

#define ALLEGRO_ROOT "allegro_wrist"
#define ALLEGRO_MOCAP_ROOT "allegro_palm"

#define OBJECT_CURRENT_POSITION "object_position"
#define OBJECT_CURRENT_ORIENTATION "object_orientation"

#define SITE_DATA_START_NAME "contact_numdata_0"
#define OBJECT_CONTACT_START_SITE_NAME "contact_site_object_0"
#define HAND_CONTACT_START_SITE_NAME "contact_site_hand_0"

#define ALLEGRO_MAX_CONTACTS 21

// #define RECORD_ALLEGRO_AGENT_DOFS

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
                : mjpc::BaseResidualFn(task), m_total_frames(0),
                  m_slowdown_factor(ALLEGRO_DEFAULT_SLOWDOWN_FACTOR),
                  m_bspline_loopback_time(0.0)
            {
                m_hand_traj_bspline_curves.clear();
                m_hand_traj_bspline_properties.clear();

                m_object_traj_bspline_curves.clear();
                m_object_traj_bspline_properties.clear();

                fill(begin(m_start_clamp_offset), end(m_start_clamp_offset), 0);
            }

            explicit ResidualFn(
                const AllegroTask *task, int total_frames,
                double slowdown_factor, string object_contact_start_data_name,
                string hand_contact_start_data_name,
                double bspline_loopback_time,
                const double start_clamp_offset[XYZ_BLOCK_SIZE],
                const vector<BSplineCurve<double> *> hand_traj_bspline_curves,
                const vector<TrajectorySplineProperties *>
                    hand_traj_bspline_properties,
                const vector<BSplineCurve<double> *> object_traj_bspline_curves,
                const vector<TrajectorySplineProperties *>
                    object_traj_bspline_properties)
                : mjpc::BaseResidualFn(task)
            {
                m_total_frames = total_frames;
                m_slowdown_factor = slowdown_factor;
                m_object_contact_start_data_name =
                    object_contact_start_data_name;
                m_hand_contact_start_data_name = hand_contact_start_data_name;

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

            // Some scenes require additional bodies inserted before the agent,
            // such as the table. This is really only necessary when we need to
            // do something "special" such as exclude collisions to force mocap
            // alignment
            int m_hand_link_body_index_offset;

            int m_total_frames;
            double m_slowdown_factor;
            string m_object_contact_start_data_name;
            string m_hand_contact_start_data_name;

            double m_bspline_loopback_time;
            double m_start_clamp_offset[XYZ_BLOCK_SIZE];

            vector<BSplineCurve<double> *> m_hand_traj_bspline_curves;
            vector<TrajectorySplineProperties *> m_hand_traj_bspline_properties;

            vector<BSplineCurve<double> *> m_object_traj_bspline_curves;
            vector<TrajectorySplineProperties *>
                m_object_traj_bspline_properties;
        };

        AllegroTask(string objectSimBodyName, string taskName,
                    string handTrajSplineFile, string objectTrajSplineFile,
                    string pcHandTrajSplineFile, double startClampOffsetX,
                    double startClampOffsetY, double startClampOffsetZ,
                    int totalFrames, string objectContactStartDataName,
                    string handContactStartDataName,
                    double slowdownFactor = ALLEGRO_DEFAULT_SLOWDOWN_FACTOR,
                    int handLinkBodyIndexOffset = 0,
                    double objectSimStartXOffset = 0.0,
                    double objectSimStartYOffset = 0.0,
                    double objectSimStartZOffset = 0.0);

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
                this, m_residual.m_total_frames, m_residual.m_slowdown_factor,
                m_residual.m_object_contact_start_data_name,
                m_residual.m_hand_contact_start_data_name,
                m_residual.m_bspline_loopback_time,
                m_residual.m_start_clamp_offset,
                m_residual.m_hand_traj_bspline_curves,
                m_residual.m_hand_traj_bspline_properties,
                m_residual.m_object_traj_bspline_curves,
                m_residual.m_object_traj_bspline_properties);
        }
        ResidualFn *InternalResidual() override { return &m_residual; }

    private:
        ResidualFn m_residual;

        string m_object_sim_body_name;
        string m_task_name;

        // Some scenes require additional bodies inserted before the agent,
        // such as the table. This is really only necessary when we need to do
        // something "special" such as exclude collisions to force mocap
        // alignment
        int m_hand_link_body_index_offset;

        // Hack to get mocap and sim body to align. Allow sim body to start
        // slightly inside table geom
        double m_object_sim_start_offset[XYZ_BLOCK_SIZE];

        int m_total_frames;
        double m_slowdown_factor;
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

        int m_failure_counter;

        int m_data_dump_write_suffix;
        vector<vector<double>> m_data_write_buffer;
    };

    class AllegroApplePassTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroApplePassTask()
            : AllegroTask("apple_sim", "apple_pass_1",
                          "mjpc/tasks/allegro/splinetrajectories/"
                          "apple_pass_1_hand.smexp",
                          "mjpc/tasks/shared_spline_trajectories/"
                          "apple_pass_1_object.smexp",
                          "mjpc/tasks/allegro/pcsplines/apple_pass_1.pcmexp",
                          -0.559059652010766, 1.009854895156828,
                          1.3654812428175624, 703,
                          "contact_pos_object_data_215_0",
                          "contact_pos_hand_data_215_0", 8.0, 0, 0, 0.012, 0)
        {
        }
    };

    class AllegroDoorknobUseTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroDoorknobUseTask()
            : AllegroTask("doorknob_sim", "doorknob_use_1",
                          "mjpc/tasks/allegro/splinetrajectories/"
                          "doorknob_use_1_hand.smexp",
                          "mjpc/tasks/shared_spline_trajectories/"
                          "doorknob_use_1_object.smexp",
                          "mjpc/tasks/allegro/pcsplines/doorknob_use_1.pcmexp",
                          -1.0543771773975556, 0.30091857905335375,
                          1.28798410204936, 1040,
                          "contact_pos_object_data_252_0",
                          "contact_pos_hand_data_252_0", 8.0)
        {
        }
    };

    class AllegroStaplerStapleTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroStaplerStapleTask()
            : AllegroTask(
                  "stapler_sim", "stapler_staple_2",
                  "mjpc/tasks/allegro/splinetrajectories/"
                  "stapler_staple_2_hand.smexp",
                  "mjpc/tasks/shared_spline_trajectories/"
                  "stapler_staple_2_object.smexp",
                  "mjpc/tasks/allegro/pcsplines/stapler_staple_2.pcmexp",
                  -0.4805667866948928, 0.58770014610545768, 1.2733766645971997,
                  877, "contact_pos_object_data_230_0",
                  "contact_pos_hand_data_230_0", 7.0, 1)
        {
        }
    };

    class AllegroWaterbottlePourTask : public AllegroTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        AllegroWaterbottlePourTask()
            : AllegroTask(
                  "waterbottle_sim", "waterbottle_pour_1",
                  "mjpc/tasks/allegro/splinetrajectories/"
                  "waterbottle_pour_1_hand.smexp",
                  "mjpc/tasks/shared_spline_trajectories/"
                  "waterbottle_pour_1_object.smexp",
                  "mjpc/tasks/allegro/pcsplines/waterbottle_pour_1.pcmexp",
                  -0.45637235839190967, 1.0530724555477113, 1.2488375856211994,
                  927, "contact_pos_object_data_185_0",
                  "contact_pos_hand_data_185_0", 5.0)
        {
        }
    };
} // namespace mjpc

#endif // MJPC_TASKS_ALLEGRO_HAND_H_
