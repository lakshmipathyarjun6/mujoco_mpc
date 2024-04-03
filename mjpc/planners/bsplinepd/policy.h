#ifndef MJPC_PLANNERS_BSPLINE_PD_POLICY_H_
#define MJPC_PLANNERS_BSPLINE_PD_POLICY_H_

#include <mujoco/mujoco.h>

#include "mjpc/planners/policy.h"
#include "mjpc/utilities.h"

using namespace std;

namespace mjpc
{

    // the policy that does nothing
    class BSplinePDPolicy : public Policy
    {
    public:
        // constructor
        BSplinePDPolicy() = default;

        // destructor
        ~BSplinePDPolicy() override = default;

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

    private:
        // assemble complete desired agent state
        vector<double> ComputeDesiredAgentState(double time) const;

        // generate spline curves from control data
        void GenerateBSplineControlData();

        // ----- members ----- //
        const mjModel *m_model;
        const Task *m_task;

        int m_num_agent_joints;
        vector<int> m_agent_joints;

        double m_root_ball_motor_kp;
        double m_root_ball_motor_kd;

        double m_intermediate_ball_motor_kp;
        double m_intermediate_ball_motor_kd;

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

#endif // MJPC_PLANNERS_BSPLINE_PD_POLICY_H_
