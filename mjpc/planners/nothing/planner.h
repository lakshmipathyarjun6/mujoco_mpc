#ifndef MJPC_PLANNERS_NOTHING_PLANNER_H_
#define MJPC_PLANNERS_NOTHING_PLANNER_H_

#include <mujoco/mujoco.h>

#include "mjpc/array_safety.h"
#include "mjpc/planners/nothing/policy.h"
#include "mjpc/planners/planner.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

using namespace std;
using namespace absl;

namespace mjpc
{
    class NothingPlanner : public Planner
    {
    public:
        // constructor
        NothingPlanner() = default;

        // destructor
        ~NothingPlanner() override = default;

        // ----- methods ----- //

        // initialize data and settings
        void Initialize(mjModel *model, const Task &task) override;

        // allocate memory
        void Allocate() override;

        // reset memory to zeros
        void Reset(int horizon,
                   const double *initial_repeated_action = nullptr) override;

        // set state
        void SetState(const State &state) override;

        // optimize nominal policy using random sampling
        void OptimizePolicy(int horizon, ThreadPool &pool) override;

        // compute trajectory using nominal policy
        void NominalTrajectory(int horizon, ThreadPool &pool) override;

        // set action from active policy
        void ActionFromPolicy(double *action, const double *state, double time,
                              bool use_previous = false) override;

        // return trajectory with best total return
        const Trajectory *BestTrajectory() override;

        // visualize planner-specific traces
        void Traces(mjvScene *scn) override;

        // planner-specific GUI elements
        void GUI(mjUI &ui) override;

        // planner-specific plots
        void Plots(mjvFigure *fig_planner, mjvFigure *fig_timer,
                   int planner_shift, int timer_shift, int planning,
                   int *shift) override;

        // return number of parameters optimized by planner
        int NumParameters() override { return 0; };

    private:
        // ----- members ----- //
        mjModel *m_model;
        const Task *m_task;

        // state
        double m_time;
        vector<double> m_state;
        vector<double> m_mocap;
        vector<double> m_userdata;

        // policies
        NothingPolicy m_active_policy; // (Guarded by mtx_)
        NothingPolicy m_candidate_policy;
        NothingPolicy m_previous_policy;

        // trajectories
        int m_num_candidate_trajectories; // actual number of candidate
                                          // trajectories
        double m_trajectory_improvement;  // improvement in total return since
                                          // last update
        Trajectory m_candidate_trajectory;

        // timing
        double m_policy_update_compute_time;

        mutable shared_mutex m_mtx;
    };
} // namespace mjpc

#endif // MJPC_PLANNERS_NOTHING_PLANNER_H_
