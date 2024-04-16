#include "mjpc/planners/pcbsplinesampling/planner.h"

namespace mjpc
{

    namespace mju = ::mujoco::util_mjpc;

    // initialize data and settings
    void PCBSplineSamplingPlanner::Initialize(mjModel *model, const Task &task)
    {
        // delete mjData instances since model might have changed.
        data_.clear();

        // allocate one mjData for nominal.
        ResizeMjData(model, 1);

        // model
        m_model = model;

        // task
        m_task = &task;

        // original bspline data
        // only care about some of the parameters - use dummies for the rest
        int bspline_degree = 0;
        int num_max_pcs = 0;
        double bspline_translation_offset[3];

        vector<double> pc_center;
        vector<double> pc_component_matrix;
        vector<DofType> bspline_doftype_data;
        vector<MeasurementUnits> bspline_measurementunit_data;

        vector<vector<double>> bspline_control_data =
            m_task->GetAgentPCBSplineControlData(
                m_bspline_dimension, bspline_degree, m_bspline_loopback_time,
                num_max_pcs, pc_center, pc_component_matrix,
                bspline_translation_offset, bspline_doftype_data,
                bspline_measurementunit_data);

        // sanity checks
        if (bspline_control_data.size() > 0 &&
            m_model->nu != bspline_control_data.size())
        {
            cout << "ERROR: Number of BSpline curves does not match number of "
                    "DOFs!"
                 << endl;
            return;
        }
        if (bspline_control_data.size() > 0)
        {
            m_num_bspline_control_points =
                bspline_control_data[0].size() / m_bspline_dimension;

            for (int i = 0; i < m_model->nu; i++)
            {
                if (bspline_control_data[i].size() / m_bspline_dimension !=
                    m_num_bspline_control_points)
                {
                    cout << "ERROR: BSplines must have same number of control "
                            "points."
                         << endl;
                    return;
                }
            }

            if (bspline_doftype_data.size() != m_model->nu)
            {
                cout << "ERROR: Each bspline must have a dof type." << endl;
                return;
            }
            if (bspline_measurementunit_data.size() != m_model->nu)
            {
                cout << "ERROR: Each bspline must have a measurement unit."
                     << endl;
                return;
            }
        }

        m_reference_control_bspline_curve = new BSplineCurve<double>(
            m_bspline_dimension, bspline_degree, m_num_bspline_control_points,
            bspline_doftype_data[0], bspline_measurementunit_data[0]);

        // set number of trajectories to rollout
        m_num_candidate_trajectories =
            GetNumberOrDefault(10, m_model, "sampling_trajectories");

        if (m_num_candidate_trajectories > kMaxTrajectory)
        {
            mju_error_i("Too many trajectories, %d is the maximum allowed.",
                        kMaxTrajectory);
        }

        m_best_candidate_trajectory_index = 0;
    }

    // allocate memory
    void PCBSplineSamplingPlanner::Allocate()
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

        m_num_active_pcs = m_active_policy.m_num_max_pcs;

        // noise
        m_noise.resize(kMaxTrajectory * m_model->nu *
                       m_num_bspline_control_points * m_bspline_dimension);

        // trajectory and parameters
        m_best_candidate_trajectory_index = -1;

        for (int i = 0; i < kMaxTrajectory; i++)
        {
            m_candidate_trajectories[i].Initialize(
                num_state, m_model->nu, m_task->num_residual, m_task->num_trace,
                kMaxTrajectoryHorizon);
            m_candidate_trajectories[i].Allocate(kMaxTrajectoryHorizon);
            m_candidate_policies[i].Allocate(m_model, *m_task,
                                             kMaxTrajectoryHorizon);
        }
    }

    // reset memory to zeros
    void PCBSplineSamplingPlanner::Reset(int horizon,
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

        // noise
        fill(m_noise.begin(), m_noise.end(), 0.0);

        // trajectory samples
        for (int i = 0; i < kMaxTrajectory; i++)
        {
            m_candidate_trajectories[i].Reset(kMaxTrajectoryHorizon);
            m_candidate_policies[i].Reset(horizon, initial_repeated_action);
        }

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

        // winner
        m_best_candidate_trajectory_index = 0;
    }

    // optimize nominal policy using random sampling
    void PCBSplineSamplingPlanner::OptimizePolicy(int horizon, ThreadPool &pool)
    {
        OptimizePolicyCandidates(1, horizon, pool);

        // ----- update policy ----- //
        // start timer
        auto policy_update_start = chrono::steady_clock::now();

        // copy best candidate to active policy
        CopyCandidateToPolicy(0);

        // improvement: compare nominal to winner
        double best_return = m_candidate_trajectories[0].total_return;
        m_trajectory_improvement = mju_max(
            best_return -
                m_candidate_trajectories[m_best_candidate_trajectory_index]
                    .total_return,
            0.0);

        // stop timer
        m_policy_update_compute_time = GetDuration(policy_update_start);
    }

    // compute trajectory using nominal policy
    void PCBSplineSamplingPlanner::NominalTrajectory(int horizon,
                                                     ThreadPool &pool)
    {
        // set policy
        auto nominal_policy =
            [&cp = m_candidate_policies[0]](double *action, const double *state,
                                            double time)
        { cp.Action(action, state, time); };

        // rollout nominal policy
        m_candidate_trajectories[0].Rollout(
            nominal_policy, m_task, m_model, data_[0].get(), m_state.data(),
            m_time, m_mocap.data(), m_userdata.data(), horizon);
    }

    // set action from policy
    void PCBSplineSamplingPlanner::ActionFromPolicy(double *action,
                                                    const double *state,
                                                    double time,
                                                    bool use_previous)
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
    const Trajectory *PCBSplineSamplingPlanner::BestTrajectory()
    {
        return m_best_candidate_trajectory_index >= 0
                   ? &m_candidate_trajectories
                         [m_best_candidate_trajectory_index]
                   : nullptr;
    }

    // set state
    void PCBSplineSamplingPlanner::SetState(const State &state)
    {
        state.CopyTo(m_state.data(), m_mocap.data(), m_userdata.data(),
                     &m_time);
    }

    // visualize planner-specific traces
    void PCBSplineSamplingPlanner::Traces(mjvScene *scn)
    {
        // sample color
        float color[4] = {1.0, 1.0, 1.0, 1.0};

        // width of a sample trace, in pixels
        double width = GetNumberOrDefault(3, m_model, "agent_sample_width");

        // scratch
        double zero3[3] = {0};
        double zero9[9] = {0};

        auto bestTrajectory = BestTrajectory();
        int num_trace = m_task->num_trace;

        // sample traces
        for (int k = 0; k < m_num_candidate_trajectories; k++)
        {
            // plot sample
            for (int i = 0; i < bestTrajectory->horizon - 1; i++)
            {
                if (scn->ngeom + num_trace > scn->maxgeom)
                {
                    cout << "Too many traces to render - breaking" << endl;
                    break;
                }

                for (int j = 0; j < num_trace; j++)
                {
                    // initialize geometry
                    mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3,
                                 zero3, zero9, color);

                    Trajectory trajectory = m_candidate_trajectories[k];

                    // make geometry
                    mjv_makeConnector(
                        &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
                        trajectory.trace[3 * num_trace * i + 3 * j],
                        trajectory.trace[3 * num_trace * i + 1 + 3 * j],
                        trajectory.trace[3 * num_trace * i + 2 + 3 * j],
                        trajectory.trace[3 * num_trace * (i + 1) + 3 * j],
                        trajectory.trace[3 * num_trace * (i + 1) + 1 + 3 * j],
                        trajectory.trace[3 * num_trace * (i + 1) + 2 + 3 * j]);

                    // increment number of geometries
                    scn->ngeom += 1;
                }
            }
        }
    }

    // planner-specific GUI elements
    void PCBSplineSamplingPlanner::GUI(mjUI &ui)
    {
        mjuiDef defPCSampling[] = {
            {mjITEM_SLIDERINT, "Samples", 2, &m_num_candidate_trajectories,
             "0 1"},
            {mjITEM_SLIDERINT, "Num PCs", 2, &m_num_active_pcs, "0 1"},
            {mjITEM_END}};

        // set number of trajectory slider limits
        mju::sprintf_arr(defPCSampling[0].other, "%i %i", 1, kMaxTrajectory);

        // set number of pc component slider limits
        mju::sprintf_arr(defPCSampling[1].other, "%i %i", 1, m_model->nu - 6);

        // add sampling planner
        mjui_add(&ui, defPCSampling);
    }

    // planner-specific plots
    void PCBSplineSamplingPlanner::Plots(mjvFigure *fig_planner,
                                         mjvFigure *fig_timer,
                                         int planner_shift, int timer_shift,
                                         int planning, int *shift)
    {
        // ----- planner ----- //
        double planner_bounds[2] = {-6.0, 6.0};

        // improvement
        mjpc::PlotUpdateData(
            fig_planner, planner_bounds,
            fig_planner->linedata[0 + planner_shift][0] + 1,
            mju_log10(mju_max(m_trajectory_improvement, 1.0e-6)), 100,
            0 + planner_shift, 0, 1, -100);

        // legend
        mju::strcpy_arr(fig_planner->linename[0 + planner_shift],
                        "Improvement");

        fig_planner->range[1][0] = planner_bounds[0];
        fig_planner->range[1][1] = planner_bounds[1];

        // bounds
        double timer_bounds[2] = {0.0, 1.0};

        // ----- timers ----- //

        PlotUpdateData(fig_timer, timer_bounds,
                       fig_timer->linedata[0 + timer_shift][0] + 1,
                       1.0e-3 * m_noise_compute_time * planning, 100,
                       0 + timer_shift, 0, 1, -100);

        PlotUpdateData(fig_timer, timer_bounds,
                       fig_timer->linedata[1 + timer_shift][0] + 1,
                       1.0e-3 * m_rollouts_compute_time * planning, 100,
                       1 + timer_shift, 0, 1, -100);

        PlotUpdateData(fig_timer, timer_bounds,
                       fig_timer->linedata[2 + timer_shift][0] + 1,
                       1.0e-3 * m_policy_update_compute_time * planning, 100,
                       2 + timer_shift, 0, 1, -100);

        // legend
        mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise");
        mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout");
        mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update");

        // planner shift
        shift[0] += 1;

        // timer shift
        shift[1] += 3;
    }

    // add random noise to candidate policy
    void PCBSplineSamplingPlanner::AddNoiseToControlPoints(
        int i, int controlPointStartIndex, int controlPointEndIndex)
    {
        // start timer
        auto noise_start = chrono::steady_clock::now();

        // sampling token
        ::BitGen gen_;

        // Reset noise
        fill(m_noise.begin(), m_noise.end(), 0.0);

        int num_dofs = m_model->nu;

        // shift index
        int shift =
            i * num_dofs * m_num_bspline_control_points * m_bspline_dimension;

        for (int dofIndex = 0; dofIndex < num_dofs; dofIndex++)
        {
            int dofOffset =
                dofIndex * m_num_bspline_control_points * m_bspline_dimension;

            double dofMult = 0.0;

            if (dofIndex < 3)
            {
                dofMult = 0.0;
            }
            else if (dofIndex < 6)
            {
                dofMult = 0.0;
            }
            else if (dofIndex > 6 + m_num_active_pcs) // do not add noise
                                                      // to non-active pcs
            {
                dofMult = 0.0;
            }
            else
            {
                dofMult = 10.0 * numbers::pi / 180.0;
            }

            for (int controlPointIndex = controlPointStartIndex;
                 controlPointIndex <= controlPointEndIndex; controlPointIndex++)
            {
                int controlPointOffset =
                    controlPointIndex * m_bspline_dimension;

                for (int dimIndex = 0; dimIndex < m_bspline_dimension;
                     dimIndex++)
                {
                    double added_noise =
                        (dimIndex == 0) ? 0.0
                                        : ::Gaussian<double>(gen_, 0.0, 1.0);

                    m_noise[shift + dofOffset + controlPointOffset + dimIndex] =
                        dofMult * added_noise;
                }
            }
        }

        m_candidate_policies[i].AdjustBSplineControlPoints(
            DataAt(m_noise, shift));

        // end timer
        IncrementAtomic(m_noise_compute_time, GetDuration(noise_start));
    }

    // compute candidate trajectories
    void PCBSplineSamplingPlanner::Rollouts(int num_trajectory, int horizon,
                                            ThreadPool &pool)
    {
        // reset noise compute time
        m_noise_compute_time = 0.0;

        int count_before = pool.GetCount();

        double current_time = fmod(m_time, m_bspline_loopback_time);
        double current_parametric_time = current_time / m_bspline_loopback_time;

        double horizon_time =
            current_time +
            horizon *
                m_model->opt.timestep; // convert discrete time to continuous
        double horizon_parametric_time = horizon_time / m_bspline_loopback_time;

        int current_time_starting_control_index;
        int current_time_ending_control_index;

        int horizon_time_starting_control_index;
        int horizon_time_ending_control_index;

        m_reference_control_bspline_curve
            ->GetContributingControlPointRangeForTime(
                current_parametric_time, current_time_starting_control_index,
                current_time_ending_control_index);

        m_reference_control_bspline_curve
            ->GetContributingControlPointRangeForTime(
                horizon_parametric_time, horizon_time_starting_control_index,
                horizon_time_ending_control_index);

        for (int i = 0; i < num_trajectory; i++)
        {
            pool.Schedule(
                [&s = *this, &model = m_model, &task = m_task, &state = m_state,
                 &time = m_time, &mocap = m_mocap, &userdata = m_userdata,
                 &start_cpi = current_time_starting_control_index,
                 &end_cpi = horizon_time_ending_control_index, horizon, i]()
                {
                    // copy nominal policy
                    {
                        const shared_lock<shared_mutex> lock(s.m_mtx);
                        s.m_candidate_policies[i]
                            .CopyControlPointsAndActivePCsFrom(
                                s.m_active_policy);
                    }

                    // sample perturbed control points
                    if (i != 0)
                    {
                        s.AddNoiseToControlPoints(i, start_cpi, end_cpi);
                    }

                    // ----- rollout sample policy ----- //

                    // policy
                    auto sample_policy_i =
                        [&candidate_policies = s.m_candidate_policies,
                         &i](double *action, const double *state, double time)
                    { candidate_policies[i].Action(action, state, time); };

                    // policy rollout
                    s.m_candidate_trajectories[i].Rollout(
                        sample_policy_i, task, model,
                        s.data_[ThreadPool::WorkerId()].get(), state.data(),
                        time, mocap.data(), userdata.data(), horizon);
                });
        }
        pool.WaitCount(count_before + num_trajectory);
        pool.ResetCount();
    }

    int PCBSplineSamplingPlanner::OptimizePolicyCandidates(int ncandidates,
                                                           int horizon,
                                                           ThreadPool &pool)
    {
        // if num_trajectory_ has changed, use it in this new iteration.
        // num_trajectory_ might change while this function runs. Keep it
        // constant for the duration of this function.
        ncandidates = min(ncandidates, m_num_candidate_trajectories);
        ResizeMjData(m_model, pool.NumThreads());

        for (int i = 0; i < kMaxTrajectory; i++)
        {
            m_candidate_policies[i].AdjustPCComponentMatrix(m_num_active_pcs);
        }

        // ----- rollout noisy policies ----- //
        // start timer
        auto rollouts_start = chrono::steady_clock::now();

        fill(m_noise.begin(), m_noise.end(), 0.0);

        // simulate and sample over policies / trajectories
        Rollouts(m_num_candidate_trajectories, horizon, pool);

        // sort candidate policies and trajectories by score
        m_candidate_trajectory_order.clear();
        m_candidate_trajectory_order.reserve(m_num_candidate_trajectories);

        for (int i = 0; i < m_num_candidate_trajectories; i++)
        {
            m_candidate_trajectory_order.push_back(i);
        }

        // sort so that the first ncandidates elements are the best candidates,
        // and the rest are in an unspecified order
        partial_sort(m_candidate_trajectory_order.begin(),
                     m_candidate_trajectory_order.begin() + ncandidates,
                     m_candidate_trajectory_order.end(),
                     [trajectory = m_candidate_trajectories](int a, int b) {
                         return trajectory[a].total_return <
                                trajectory[b].total_return;
                     });

        // stop timer
        m_rollouts_compute_time = GetDuration(rollouts_start);

        return ncandidates;
    }

    double PCBSplineSamplingPlanner::CandidateScore(int candidate) const
    {
        return m_candidate_trajectories[m_candidate_trajectory_order[candidate]]
            .total_return;
    }

    // set action from candidate policy
    void PCBSplineSamplingPlanner::ActionFromCandidatePolicy(
        double *action, int candidate, const double *state, double time)
    {
        m_candidate_policies[m_candidate_trajectory_order[candidate]].Action(
            action, state, time);
    }

    void PCBSplineSamplingPlanner::CopyCandidateToPolicy(int candidate)
    {
        // set winner
        m_best_candidate_trajectory_index =
            m_candidate_trajectory_order[candidate];

        {
            const shared_lock<shared_mutex> lock(m_mtx);
            m_previous_policy = m_active_policy;
            m_active_policy =
                m_candidate_policies[m_best_candidate_trajectory_index];
        }
    }

} // namespace mjpc
