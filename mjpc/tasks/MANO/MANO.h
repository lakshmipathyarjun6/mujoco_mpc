#ifndef MJPC_TASKS_MANO_HAND_H_
#define MJPC_TASKS_MANO_HAND_H_

#include <mujoco/mujoco.h>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

using namespace std;

namespace mjpc
{

    class MANOTask : public Task
    {
    public:
        class ResidualFn : public mjpc::BaseResidualFn
        {
        public:
            explicit ResidualFn(const MANOTask *task)
                : mjpc::BaseResidualFn(task)
            {
            }

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            friend class MANOTask;
        };

        MANOTask(string objectSimBodyName);

        vector<double> GetDesiredState(double time) const override;

        vector<double> GetDesiredStateFromPCs(double time) const override;

        vector<vector<double>> GetBSplineControlData(
            int &dimension, int &degree, double &loopbackTime,
            double translationOffset[3], vector<DofType> &dofTypes,
            vector<MeasurementUnits> &measurementUnits) const override;

        // --------------------- Transition for allegro task
        // ------------------------
        //   Set `data->mocap_pos` based on `data->time` to move the object
        //   site.
        // ---------------------------------------------------------------------------
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        unique_ptr<mjpc::ResidualFn> ResidualLocked() const override
        {
            return make_unique<ResidualFn>(this);
        }
        ResidualFn *InternalResidual() override { return &m_residual; }

    private:
        ResidualFn m_residual;

        string m_object_sim_body_name;
    };

    class MANOAppleTask : public MANOTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        MANOAppleTask() : MANOTask("apple_sim") {}

    private:
    };

    class MANODoorknobTask : public MANOTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        MANODoorknobTask() : MANOTask("doorknob_sim") {}
    };

} // namespace mjpc

#endif // MJPC_TASKS_MANO_HAND_H_
