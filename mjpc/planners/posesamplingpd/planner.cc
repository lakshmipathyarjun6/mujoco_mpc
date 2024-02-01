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

        double rounded_index = floor(time * 12);
        int current_index = int(rounded_index) % model->nkey;

        int handMocapQOffset = model->nq * current_index;

        double posError[22];
        double velError[22];

        mju_sub(posError, model->key_qpos + handMocapQOffset, state, model->nu);
        mju_scl(posError, posError, 20, 3);
        mju_scl(posError + 3, posError + 3, 10, 3);
        mju_scl(posError + 6, posError + 6, 5, model->nu - 6);

        mju_copy(velError, state + model->nq, model->nu); // want velocity close to 0
        mju_scl(velError, velError, 1, 3);
        mju_scl(velError + 3, velError + 3, 0, model->nu - 3);

        mju_sub(action, posError, velError, model->nu);

        // Clamp controls
        Clamp(action, model->actuator_ctrlrange, model->nu);
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
