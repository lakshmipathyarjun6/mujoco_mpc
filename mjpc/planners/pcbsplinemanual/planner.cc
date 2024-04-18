#include "mjpc/planners/pcbsplinemanual/planner.h"

namespace mjpc
{

    namespace mju = ::mujoco::util_mjpc;

    // initialize data and settings
    void PCBSplineManualPlanner::Initialize(mjModel *model, const Task &task)
    {
        // delete mjData instances since model might have changed.
        data_.clear();

        // allocate one mjData for nominal.
        ResizeMjData(model, 1);

        // model
        m_model = model;

        // task
        m_task = &task;
    }

    // allocate memory
    void PCBSplineManualPlanner::Allocate()
    {
        // initial state
        int num_state = m_model->nq + m_model->nv + m_model->na;

        // state
        m_state.resize(num_state);
        m_mocap.resize(7 * m_model->nmocap);
        m_userdata.resize(m_model->nuserdata);

        // policy
        m_active_policy.Allocate(m_model, *m_task, kMaxTrajectoryHorizon);

        m_trajectory.Initialize(num_state, m_model->nu, m_task->num_residual,
                                m_task->num_trace, kMaxTrajectoryHorizon);
        m_trajectory.Allocate(kMaxTrajectoryHorizon);
    }

    // reset memory to zeros
    void PCBSplineManualPlanner::Reset(int horizon,
                                       const double *initial_repeated_action)
    {
        // state
        fill(m_state.begin(), m_state.end(), 0.0);
        fill(m_mocap.begin(), m_mocap.end(), 0.0);
        fill(m_userdata.begin(), m_userdata.end(), 0.0);
        m_time = 0.0;

        // policy parameters
        m_active_policy.Reset(horizon, initial_repeated_action);

        // trajectory samples
        m_trajectory.Reset(kMaxTrajectoryHorizon);

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
    }

    // optimize nominal policy using random sampling
    void PCBSplineManualPlanner::OptimizePolicy(int horizon, ThreadPool &pool)
    {
        // Do bsplinepd
    }

    // compute trajectory using nominal policy
    void PCBSplineManualPlanner::NominalTrajectory(int horizon,
                                                   ThreadPool &pool)
    {
        // set policy
        auto nominal_policy = [&cp = m_active_policy](double *action,
                                                      const double *state,
                                                      double time)
        { cp.Action(action, state, time); };

        // rollout nominal policy
        m_trajectory.Rollout(nominal_policy, m_task, m_model, data_[0].get(),
                             m_state.data(), m_time, m_mocap.data(),
                             m_userdata.data(), horizon);
    }

    // set action from policy
    void PCBSplineManualPlanner::ActionFromPolicy(double *action,
                                                  const double *state,
                                                  double time,
                                                  bool use_previous)
    {
        const shared_lock<shared_mutex> lock(m_mtx);
        m_active_policy.Action(action, state, time);
    }

    // return trajectory with best total return
    const Trajectory *PCBSplineManualPlanner::BestTrajectory()
    {
        return &m_trajectory;
    }

    // set state
    void PCBSplineManualPlanner::SetState(const State &state)
    {
        state.CopyTo(m_state.data(), m_mocap.data(), m_userdata.data(),
                     &m_time);
    }

    // visualize planner-specific traces
    void PCBSplineManualPlanner::Traces(mjvScene *scn) {}

    // planner-specific GUI elements
    void PCBSplineManualPlanner::GUI(mjUI &ui)
    {
        int num_pcs = m_active_policy.m_num_pcs;

        vector<mjuiDef> defPCBsplineManualPlanner;

        for (int i = 0; i < num_pcs; i++)
        {
            char numBuffer[3];
            snprintf(numBuffer, 3, "%02d", i + 1);

            mjuiDef element = {mjITEM_SLIDERNUM,
                               {'P', 'C', ' ', numBuffer[0], numBuffer[1]},
                               2,
                               &m_active_policy.m_pc_state[i],
                               "-3 3"};

            defPCBsplineManualPlanner.push_back(element);
        }
        defPCBsplineManualPlanner.push_back({mjITEM_END});

        // add gradient descent planner
        mjui_add(&ui, defPCBsplineManualPlanner.data());
    }

    // planner-specific plots
    void PCBSplineManualPlanner::Plots(mjvFigure *fig_planner,
                                       mjvFigure *fig_timer, int planner_shift,
                                       int timer_shift, int planning,
                                       int *shift)
    {
    }
} // namespace mjpc
