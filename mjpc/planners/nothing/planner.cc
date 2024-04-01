#include "mjpc/planners/nothing/planner.h"

namespace mjpc
{

    namespace mju = ::mujoco::util_mjpc;

    // initialize data and settings
    void NothingPlanner::Initialize(mjModel *model, const Task &task)
    {
        // delete mjData instances since model might have changed.
        data_.clear();

        // allocate one mjData for nominal.
        ResizeMjData(model, 1);

        // model
        m_model = model;

        // task
        m_task = &task;

        // set number of trajectories to rollout
        m_num_candidate_trajectories = 1;
    }

    // allocate memory
    void NothingPlanner::Allocate()
    {
        // initial state
        int num_state = m_model->nq + m_model->nv + m_model->na;

        // state
        m_state.resize(num_state);
        m_mocap.resize(7 * m_model->nmocap);
        m_userdata.resize(m_model->nuserdata);

        // policy
        m_active_policy.Allocate(m_model, *m_task, kMaxTrajectoryHorizon);
        m_previous_policy.Allocate(m_model, *m_task, kMaxTrajectoryHorizon);

        m_candidate_trajectory.Initialize(
            num_state, m_model->nu, m_task->num_residual, m_task->num_trace,
            kMaxTrajectoryHorizon);
        m_candidate_trajectory.Allocate(kMaxTrajectoryHorizon);
        m_candidate_policy.Allocate(m_model, *m_task, kMaxTrajectoryHorizon);
    }

    // reset memory to zeros
    void NothingPlanner::Reset(int horizon,
                               const double *initial_repeated_action)
    {
        // state
        fill(m_state.begin(), m_state.end(), 0.0);
        fill(m_mocap.begin(), m_mocap.end(), 0.0);
        fill(m_userdata.begin(), m_userdata.end(), 0.0);
        m_time = 0.0;

        // policy parameters
        m_active_policy.Reset(horizon, initial_repeated_action);
        m_previous_policy.Reset(horizon, initial_repeated_action);

        // trajectory samples
        m_candidate_trajectory.Reset(kMaxTrajectoryHorizon);
        m_candidate_policy.Reset(horizon, initial_repeated_action);

        for (const auto &d : data_)
        {
            if (initial_repeated_action)
            {
                mju_copy(d->ctrl, initial_repeated_action, m_model->nu);
            }
            else
            {
                mju_zero(d->ctrl, m_model->nu);
            }
        }

        // improvement
        m_trajectory_improvement = 0.0;
    }

    // optimize nominal policy using random sampling
    void NothingPlanner::OptimizePolicy(int horizon, ThreadPool &pool)
    {
        // ----- update policy ----- //
        // start timer
        auto policy_update_start = chrono::steady_clock::now();

        // improvement: compare nominal to winner
        double best_return = m_candidate_trajectory.total_return;
        m_trajectory_improvement =
            mju_max(best_return - m_candidate_trajectory.total_return, 0.0);

        // stop timer
        m_policy_update_compute_time = GetDuration(policy_update_start);
    }

    // compute trajectory using nominal policy
    void NothingPlanner::NominalTrajectory(int horizon, ThreadPool &pool)
    {
        // set policy
        auto nominal_policy = [&cp = m_candidate_policy](double *action,
                                                         const double *state,
                                                         double time)
        { cp.Action(action, state, time); };

        // rollout nominal policy
        m_candidate_trajectory.Rollout(
            nominal_policy, m_task, m_model, data_[0].get(), m_state.data(),
            m_time, m_mocap.data(), m_userdata.data(), horizon);
    }

    // set action from policy
    void NothingPlanner::ActionFromPolicy(double *action, const double *state,
                                          double time, bool use_previous)
    {
        const shared_lock<shared_mutex> lock(m_mtx);

        if (use_previous)
        {
            m_previous_policy.Action(action, state, time);
        }
        else
        {
            m_active_policy.Action(action, state, time);
        }
    }

    // return trajectory with best total return
    const Trajectory *NothingPlanner::BestTrajectory()
    {
        return &m_candidate_trajectory;
    }

    // set state
    void NothingPlanner::SetState(const State &state)
    {
        state.CopyTo(m_state.data(), m_mocap.data(), m_userdata.data(),
                     &m_time);

        if (m_task->mode == 0)
        {
            Reset(0, nullptr);
        }
    }

    // visualize planner-specific traces
    void NothingPlanner::Traces(mjvScene *scn) {}

    // planner-specific GUI elements
    void NothingPlanner::GUI(mjUI &ui) {}

    // planner-specific plots
    void NothingPlanner::Plots(mjvFigure *fig_planner, mjvFigure *fig_timer,
                               int planner_shift, int timer_shift, int planning,
                               int *shift)
    {
    }

} // namespace mjpc
