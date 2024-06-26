#ifndef MJPC_PLANNERS_BSPLINE_SAMPLING_PLANNER_H_
#define MJPC_PLANNERS_BSPLINE_SAMPLING_PLANNER_H_

#include <mujoco/mujoco.h>

#include <atomic>
#include <shared_mutex>
#include <vector>

#include <absl/random/random.h>

#include "mjpc/array_safety.h"
#include "mjpc/planners/bsplinesampling/policy.h"
#include "mjpc/planners/planner.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

using namespace std;
using namespace absl;

namespace mjpc
{
    class BSplineSamplingPlanner : public RankedPlanner
    {
    public:
        // constructor
        BSplineSamplingPlanner() = default;

        // destructor
        ~BSplineSamplingPlanner() override = default;

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
        int NumParameters() override { return 4 * m_model->nu; };

        // add noise to nominal policy
        void AddNoiseToControlPoints(int i, int controlPointStartIndex,
                                     int controlPointEndIndex);

        // compute candidate trajectories
        void Rollouts(int num_trajectory, int horizon, ThreadPool &pool);

        // optimizes policies, but rather than picking the best, generate up to
        // ncandidates. returns number of candidates created.
        int OptimizePolicyCandidates(int ncandidates, int horizon,
                                     ThreadPool &pool) override;

        // returns the total return for the nth candidate (or another score to
        // minimize)
        double CandidateScore(int candidate) const override;

        // set action from candidate policy
        void ActionFromCandidatePolicy(double *action, int candidate,
                                       const double *state,
                                       double time) override;

        void CopyCandidateToPolicy(int candidate) override;

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
        BSplineSamplingPolicy m_active_policy; // (Guarded by mtx_)
        BSplineSamplingPolicy m_candidate_policies[kMaxTrajectory];
        BSplineSamplingPolicy m_previous_policy;

        // trajectories
        int m_num_candidate_trajectories;      // actual number of candidate
                                               // trajectories
        int m_best_candidate_trajectory_index; // best trajectory index in
                                               // unordered trajectory list
        double m_trajectory_improvement; // improvement in total return since
                                         // last update
        Trajectory m_candidate_trajectories[kMaxTrajectory]; // allocate maximum
                                                             // trajectory space
        vector<int> m_candidate_trajectory_order; // order of indices of rolled
                                                  // out trajectories, ordered
                                                  // by total return

        // ----- noise ----- //
        vector<double> m_noise;

        // timing
        atomic<double> m_noise_compute_time;
        double m_rollouts_compute_time;
        double m_policy_update_compute_time;

        // bspline parameters that we actually care about
        int m_num_bspline_control_points;
        int m_bspline_dimension;
        double m_bspline_loopback_time;
        BSplineCurve<double> *m_reference_control_bspline_curve;

        mutable shared_mutex m_mtx;
    };
} // namespace mjpc

#endif // MJPC_PLANNERS_BSPLINE_SAMPLING_PLANNER_H_
