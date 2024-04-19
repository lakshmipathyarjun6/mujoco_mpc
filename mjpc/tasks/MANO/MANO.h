#ifndef MJPC_TASKS_MANO_HAND_H_
#define MJPC_TASKS_MANO_HAND_H_

#include <mujoco/mujoco.h>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

#include "JSONUtils.hpp"

#define DEFAULT_MOCAP_FPS 120
#define MANO_SLOWDOWN_FACTOR 10

// Not equal due to ball joints
#define MANO_DOFS 67
#define MANO_VEL_DOFS 51
#define MANO_NON_ROOT_VEL_DOFS 45

#define XYZ_BLOCK_SIZE 3

#define MANO_ROOT "wrist"

#define OBJECT_CURRENT_POSITION "object_position"
#define OBJECT_CURRENT_ORIENTATION "object_orientation"

#define SITE_DATA_START_NAME "contact_numdata_0"
#define OBJECT_CONTACT_START_SITE_NAME "contact_site_object_0"
#define HAND_CONTACT_START_SITE_NAME "contact_site_hand_0"

#define MAX_CONTACTS 16

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
                : mjpc::BaseResidualFn(task), m_num_active_contacts(0)
            {
                fill(begin(m_r_object_mocap_pos_buffer),
                     end(m_r_object_mocap_pos_buffer), 0);
                fill(begin(m_r_object_mocap_quat_buffer),
                     end(m_r_object_mocap_quat_buffer), 0);
                fill(begin(m_r_hand_contact_position_buffer),
                     end(m_r_hand_contact_position_buffer), 0);
                fill(begin(m_r_object_contact_position_buffer),
                     end(m_r_hand_contact_position_buffer), 0);
            }

            explicit ResidualFn(
                const MANOTask *task, int num_active_contacts,
                const double mocap_object_pos_buffer[3],
                const double mocap_object_quat_buffer[4],
                const double
                    hand_contact_position_buffer[MAX_CONTACTS * XYZ_BLOCK_SIZE],
                const double object_contact_position_buffer[MAX_CONTACTS *
                                                            XYZ_BLOCK_SIZE])
                : mjpc::BaseResidualFn(task)
            {
                m_num_active_contacts = num_active_contacts;

                mju_copy3(m_r_object_mocap_pos_buffer, mocap_object_pos_buffer);
                mju_copy4(m_r_object_mocap_quat_buffer,
                          mocap_object_quat_buffer);
                mju_copy(m_r_hand_contact_position_buffer,
                         hand_contact_position_buffer,
                         MAX_CONTACTS * XYZ_BLOCK_SIZE);
                mju_copy(m_r_object_contact_position_buffer,
                         object_contact_position_buffer,
                         MAX_CONTACTS * XYZ_BLOCK_SIZE);
            }

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            friend class MANOTask;

            double m_r_object_mocap_pos_buffer[3];
            double m_r_object_mocap_quat_buffer[4];

            int m_num_active_contacts;

            double
                m_r_hand_contact_position_buffer[MAX_CONTACTS * XYZ_BLOCK_SIZE];
            double m_r_object_contact_position_buffer[MAX_CONTACTS *
                                                      XYZ_BLOCK_SIZE];
        };

        MANOTask(string objectSimBodyName, string handTrajSplineFile,
                 string objectTrajSplineFile, string pcHandTrajSplineFile,
                 double startClampOffsetX, double startClampOffsetY,
                 double startClampOffsetZ, int totalFrames,
                 string objectContactStartDataName,
                 string handContactStartDataName);

        vector<double> GetDesiredAgentState(double time) const;

        vector<double> GetDesiredAgentStateFromPCs(double time) const;

        vector<double> GetDesiredObjectState(double time) const;

        vector<vector<double>> GetAgentBSplineControlData(
            int &dimension, int &degree, double &loopbackTime,
            double translationOffset[3], vector<DofType> &dofTypes,
            vector<MeasurementUnits> &measurementUnits) const override;

        vector<vector<double>> GetAgentPCBSplineControlData(
            int &dimension, int &degree, double &loopbackTime, int &numMaxPCs,
            vector<double> &centerData, vector<double> &componentData,
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
            return make_unique<ResidualFn>(
                this, m_residual.m_num_active_contacts,
                m_residual.m_r_object_mocap_pos_buffer,
                m_residual.m_r_object_mocap_quat_buffer,
                m_residual.m_r_hand_contact_position_buffer,
                m_residual.m_r_object_contact_position_buffer);
        }
        ResidualFn *InternalResidual() override { return &m_residual; }

    private:
        ResidualFn m_residual;

        string m_object_sim_body_name;

        int m_total_frames;
        string m_object_contact_start_data_name;
        string m_hand_contact_start_data_name;

        double m_hand_kinematic_buffer[MANO_DOFS];

        int m_spline_dimension;
        int m_spline_degree;

        vector<BSplineCurve<double> *> m_hand_traj_bspline_curves;
        vector<TrajectorySplineProperties> m_hand_traj_bspline_properties;

        vector<BSplineCurve<double> *> m_object_traj_bspline_curves;
        vector<TrajectorySplineProperties> m_object_traj_bspline_properties;

        int m_num_pcs;
        vector<double> m_hand_pc_center;
        vector<double> m_hand_pc_component_matrix;
        vector<BSplineCurve<double> *> m_hand_pc_traj_bspline_curves;
        vector<TrajectorySplineProperties> m_hand_pc_traj_bspline_properties;

        double m_spline_loopback_time;
        double m_start_clamp_offset[3];

        map<string, DofType> m_doftype_property_mappings;
        map<string, MeasurementUnits> m_measurement_units_property_mappings;
    };

    class MANOApplePassTask : public MANOTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        MANOApplePassTask()
            : MANOTask("apple_sim",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                       "splinetrajectories/apple_pass_1_hand.smexp",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/"
                       "shared_spline_trajectories/apple_pass_1_object.smexp",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                       "pcsplines/apple_pass_1.pcmexp",
                       -0.58147233724594119, 1.0124462842941284,
                       1.3647385835647584, 703, "contact_pos_object_data_215_0",
                       "contact_pos_hand_data_215_0")
        {
        }

    private:
    };

    class MANODoorknobUseTask : public MANOTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        MANODoorknobUseTask()
            : MANOTask("doorknob_sim",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                       "splinetrajectories/doorknob_use_1_hand.smexp",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/"
                       "shared_spline_trajectories/doorknob_use_1_object.smexp",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                       "pcsplines/doorknob_use_1.pcmexp",
                       -1.0741884708404541, 0.31418800354003908,
                       1.298376441001892, 1040, "contact_pos_object_data_252_0",
                       "contact_pos_hand_data_252_0")
        {
        }
    };

    class MANOFlashlightOnTask : public MANOTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        MANOFlashlightOnTask()
            : MANOTask(
                  "flashlight_sim",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                  "splinetrajectories/flashlight_on_1_hand.smexp",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/"
                  "shared_spline_trajectories/flashlight_on_1_object.smexp",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                  "pcsplines/flashlight_on_1.pcmexp",
                  -0.6258119344711304, 0.8344507813453675, 1.3911676406860352,
                  1040, "contact_pos_object_data_205_0",
                  "contact_pos_hand_data_205_0")
        {
        }
    };

    class MANOHammerUseTask : public MANOTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        MANOHammerUseTask()
            : MANOTask("hammer_sim",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                       "splinetrajectories/hammer_use_2_hand.smexp",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/"
                       "shared_spline_trajectories/hammer_use_2_object.smexp",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                       "pcsplines/hammer_use_2.pcmexp",
                       -0.7610342502593994, 0.60684651136398318,
                       1.355204939842224, 768, "contact_pos_object_data_139_0",
                       "contact_pos_hand_data_139_0")
        {
        }
    };

    class MANOPhoneCallTask : public MANOTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        MANOPhoneCallTask()
            : MANOTask("phone_sim",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                       "splinetrajectories/phone_call_1_hand.smexp",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/"
                       "shared_spline_trajectories/phone_call_1_object.smexp",
                       "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                       "pcsplines/phone_call_1.pcmexp",
                       -0.7065898180007935, 0.3405895233154297,
                       1.313579797744751, 1145, "contact_pos_object_data_198_0",
                       "contact_pos_hand_data_198_0")
        {
        }
    };

    class MANOStaplerStapleTask : public MANOTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        MANOStaplerStapleTask()
            : MANOTask(
                  "stapler_sim",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                  "splinetrajectories/stapler_staple_2_hand.smexp",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/"
                  "shared_spline_trajectories/stapler_staple_2_object.smexp",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                  "pcsplines/stapler_staple_2.pcmexp",
                  -0.4984360337257385, 0.5916348099708557, 1.2731690406799317,
                  877, "contact_pos_object_data_230_0",
                  "contact_pos_hand_data_230_0")
        {
        }
    };

    class MANOWaterbottlePourTask : public MANOTask
    {
    public:
        string Name() const override;
        string XmlPath() const override;

        MANOWaterbottlePourTask()
            : MANOTask(
                  "waterbottle_sim",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                  "splinetrajectories/waterbottle_pour_1_hand.smexp",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/"
                  "shared_spline_trajectories/waterbottle_pour_1_object.smexp",
                  "/Users/arjunl/mujoco_mpc/mjpc/tasks/MANO/"
                  "pcsplines/waterbottle_pour_1.pcmexp",
                  -0.4804867506027222, 1.0599700212478638, 1.2646256685256958,
                  927, "contact_pos_object_data_185_0",
                  "contact_pos_hand_data_185_0")
        {
        }
    };

} // namespace mjpc

#endif // MJPC_TASKS_MANO_HAND_H_
