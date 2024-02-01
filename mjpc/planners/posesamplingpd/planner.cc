#include "mjpc/planners/posesamplingpd/planner.h"

namespace mjpc
{

    namespace mju = ::mujoco::util_mjpc;

    // initialize data and settings
    void PoseSamplingPDPlanner::Initialize(mjModel *model, const Task &task)
    {
        // delete mjData instances since model might have changed.
        data_.clear();

        // allocate one mjData for nominal.
        ResizeMjData(model, 1);

        // model
        this->model = model;

        // task
        this->task = &task;
    }

    // allocate memory
    void PoseSamplingPDPlanner::Allocate()
    {
        // initial state
        int num_state = model->nq + model->nv + model->na;

        // state
        state.resize(num_state);
        mocap.resize(7 * model->nmocap);
        userdata.resize(model->nuserdata);

        // policy
        policy.Allocate(model, *task, kMaxTrajectoryHorizon);
    }

    // reset memory to zeros
    void PoseSamplingPDPlanner::Reset(int horizon,
                                      const double *initial_repeated_action)
    {
        // state
        fill(state.begin(), state.end(), 0.0);
        fill(mocap.begin(), mocap.end(), 0.0);
        fill(userdata.begin(), userdata.end(), 0.0);
        time = 0.0;
    }

    // optimize nominal policy using random sampling
    void PoseSamplingPDPlanner::OptimizePolicy(int horizon, ThreadPool &pool)
    {
    }

    // compute trajectory using nominal policy
    void PoseSamplingPDPlanner::NominalTrajectory(int horizon, ThreadPool &pool)
    {
    }

    // set action from policy
    void PoseSamplingPDPlanner::ActionFromPolicy(double *action, const double *state,
                                                 double time, bool use_previous)
    {
        const shared_lock<shared_mutex> lock(mtx_);
        policy.Action(action, state, time);
    }

    // return trajectory with best total return
    const Trajectory *PoseSamplingPDPlanner::BestTrajectory()
    {
        return nullptr;
    }

    // set state
    void PoseSamplingPDPlanner::SetState(const State &state)
    {
        state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
                     &this->time);
    }

    // visualize planner-specific traces
    void PoseSamplingPDPlanner::Traces(mjvScene *scn)
    {
    }

    // planner-specific GUI elements
    void PoseSamplingPDPlanner::GUI(mjUI &ui)
    {
    }

    // planner-specific plots
    void PoseSamplingPDPlanner::Plots(mjvFigure *fig_planner, mjvFigure *fig_timer,
                                      int planner_shift, int timer_shift, int planning,
                                      int *shift)
    {
    }

}
