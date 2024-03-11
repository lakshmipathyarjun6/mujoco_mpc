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

#ifndef MJPC_PLANNERS_POSE_SAMPLING_PD_POLICY_H_
#define MJPC_PLANNERS_POSE_SAMPLING_PD_POLICY_H_

#include <mujoco/mujoco.h>

#include "mjpc/planners/policy.h"
#include "mjpc/spline/bspline.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

#include <set>

using namespace std;

namespace mjpc
{

    // policy for sampling planner
    class PoseSamplingPDPolicy : public Policy
    {
    public:
        // constructor
        PoseSamplingPDPolicy() = default;

        // destructor
        ~PoseSamplingPDPolicy() override = default;

        // ----- methods ----- //

        // allocate memory
        void Allocate(const mjModel *model, const Task &task,
                      int horizon) override;

        // reset memory to zeros and reference configs to original mocap
        // trajectory
        void Reset(int horizon,
                   const double *initial_repeated_action = nullptr) override;

        // set action from policy
        void Action(double *action, const double *state,
                    double time) const override;

        // copy parameters
        void
        CopyReferenceConfigsFrom(const vector<double> &src_reference_configs);

        // generate splien curves from control data
        void GenerateBSplineControlData();

        vector<double> m_reference_configs;

    private:
        // ----- members ----- //
        const mjModel *m_model;
        const Task *m_task;

        double m_ball_motor_kp;
        double m_ball_motor_kd;

        int m_num_bspline_control_points;
        int m_bspline_dimension;
        int m_bspline_degree;
        double m_bspline_loopback_time;
        double m_bspline_translation_offset[3];

        vector<vector<double>> m_bspline_control_data;
        vector<DofType> m_bspline_doftype_data;
        vector<MeasurementUnits> m_bspline_measurementunit_data;

        vector<BSplineCurve<double> *> m_control_bspline_curves;
    };

} // namespace mjpc

#endif // MJPC_PLANNERS_POSE_SAMPLING_PD_POLICY_H_
