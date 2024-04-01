#include "mjpc/tasks/MANO/MANO.h"

namespace mjpc
{
    // ---------- Residuals for MANO hand manipulation task ---------
    //   Number of residuals: 4
    //     Residual (0): object_position - object_traj_position
    //     Residual (1): object_orientation - object_traj_orientation
    //     Residual (2): hand joint velocity
    //     Residual (3): contact alignment
    // ------------------------------------------------------------

    // NOTE: Currently unclear how to adapt to non-free objects (e.g. doorknob)
    void MANOTask::ResidualFn::Residual(const mjModel *model,
                                        const mjData *data,
                                        double *residual) const
    {
        int offset = 0;

        // ---------- Residual (0) ----------
        // position error
        mju_zero3(residual + offset);
        offset += 3;

        // ---------- Residual (1) ----------
        // orientation error
        mju_zero3(residual + offset);
        offset += 3;

        // sensor dim sanity check
        CheckSensorDim(model, offset);
    }

    // --------------------- Transition for MANO task
    // ------------------------
    //   Set `data->mocap_pos` based on `data->time` to move the object site.
    // ---------------------------------------------------------------------------
    void MANOTask::TransitionLocked(mjModel *model, mjData *data) {}

    MANOTask::MANOTask(string objectSimBodyName)
        : m_residual(this), m_object_sim_body_name(objectSimBodyName)
    {
    }

    vector<double> MANOTask::GetDesiredState(double time) const
    {
        vector<double> desiredState;
        return desiredState;
    }

    vector<double> MANOTask::GetDesiredStateFromPCs(double time) const
    {
        vector<double> desiredState;
        return desiredState;
    }

    vector<vector<double>> MANOTask::GetBSplineControlData(
        int &dimension, int &degree, double &loopbackTime,
        double translationOffset[3], vector<DofType> &dofTypes,
        vector<MeasurementUnits> &measurementUnits) const
    {
        vector<vector<double>> bsplineControlData;
        return bsplineControlData;
    }

    string MANOAppleTask::XmlPath() const
    {
        return GetModelPath("MANO/task_apple.xml");
    }

    string MANOAppleTask::Name() const { return "MANO Apple Pass"; }

    string MANODoorknobTask::XmlPath() const
    {
        return GetModelPath("MANO/task_doorknob.xml");
    }

    string MANODoorknobTask::Name() const { return "MANO Doorknob Use"; }

} // namespace mjpc
