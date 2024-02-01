#ifndef MJPC_PLANNERS_POSE_SAMPLING_PD_PLANNER_H_
#define MJPC_PLANNERS_POSE_SAMPLING_PD_PLANNER_H_

#include <mujoco/mujoco.h>

#include <atomic>
#include <shared_mutex>
#include <vector>

#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

using namespace std;

namespace mjpc
{

    class PoseSamplingPDPlanner : public Planner
    {
    public:
        // constructor
        PoseSamplingPDPlanner() = default;

        // destructor
        ~PoseSamplingPDPlanner() override = default;

        // ----- methods ----- //

        // initialize data and settings
        void Initialize(mjModel *model, const Task &task) override;

        // allocate memory
        void Allocate() override;

        // reset memory to zeros
        void Reset(int horizon, const double *initial_repeated_action = nullptr) override;

        // set state
        void SetState(const State &state) override;

        // optimize nominal policy using random sampling
        void OptimizePolicy(int horizon, ThreadPool &pool) override;

        // compute trajectory using nominal policy
        void NominalTrajectory(int horizon, ThreadPool &pool) override;

        // set action from policy
        void ActionFromPolicy(double *action, const double *state,
                              double time, bool use_previous = false) override;

        // return trajectory with best total return
        const Trajectory *BestTrajectory() override;

        // visualize planner-specific traces
        void Traces(mjvScene *scn) override;

        // planner-specific GUI elements
        void GUI(mjUI &ui) override;

        // planner-specific plots
        void Plots(mjvFigure *fig_planner, mjvFigure *fig_timer, int planner_shift,
                   int timer_shift, int planning, int *shift) override;

        // ----- members ----- //
        mjModel *model;
        const Task *task;

        // state
        vector<double> state;
        double time;
        vector<double> mocap;
        vector<double> userdata;

        mutable shared_mutex mtx_;
    };
}

#endif // MJPC_PLANNERS_POSE_SAMPLING_PD_PLANNER_H_
