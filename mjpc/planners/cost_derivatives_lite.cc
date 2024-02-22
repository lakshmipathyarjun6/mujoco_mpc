// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/planners/cost_derivatives_lite.h"

namespace mjpc
{
    // allocate memory
    void CostDerivativesLite::Allocate(int dim_state_derivative, int dim_action,
                                       int dim_residual, int T, int dim_max)
    {
        // norm derivatives
        cr.resize(dim_residual * T);

        // cost gradients
        cx.resize(dim_state_derivative * T);
        cu.resize(dim_action * T);

        // scratch space
        c_scratch_.resize(T * dim_max * dim_max);
        cx_scratch_.resize(T * dim_state_derivative);
        cu_scratch_.resize(T * dim_action);
    }

    // reset memory to zeros
    void CostDerivativesLite::Reset(int dim_state_derivative, int dim_action,
                                    int dim_residual, int T)
    {
        fill(cr.begin(), cr.begin() + dim_residual * T, 0.0);
        fill(cx.begin(), cx.begin() + dim_state_derivative * T, 0.0);
        fill(cu.begin(), cu.begin() + dim_action * T, 0.0);
        fill(cx_scratch_.begin(), cx_scratch_.begin() + T * dim_state_derivative,
             0.0);
        fill(cu_scratch_.begin(), cu_scratch_.begin() + T * dim_action, 0.0);
    }

    // compute derivatives at one time step
    double CostDerivativesLite::DerivativeStep(
        double *Cx, double *Cu,
        double *Cr, double *C_scratch,
        double *Cx_scratch, double *Cu_scratch,
        const double *r, const double *rx,
        const double *ru, int nr, int nx, int dim_action,
        double weight, const double *p, NormType type)
    {
        // norm derivatives
        double C = Norm(Cr, nullptr, r, p, nr, type);

        // cx
        mju_mulMatTVec(Cx_scratch, rx, Cr, nr, nx);
        mju_addToScl(Cx, Cx_scratch, weight, nx);

        // cu
        mju_mulMatTVec(Cu_scratch, ru, Cr, nr, dim_action);
        mju_addToScl(Cu, Cu_scratch, weight, dim_action);

        return weight * C;
    }

    // compute derivatives at all time steps
    void CostDerivativesLite::Compute(double *r, double *rx, double *ru,
                                      int dim_state_derivative, int dim_action,
                                      int dim_max, int num_sensors, int num_residual,
                                      const int *dim_norm_residual, int num_term,
                                      const double *weights, const NormType *norms,
                                      const double *parameters,
                                      const int *num_norm_parameter, double risk,
                                      int T, ThreadPool &pool)
    {
        // reset
        this->Reset(dim_state_derivative, dim_action, num_residual, T);
        {
            int count_before = pool.GetCount();
            for (int t = 0; t < T; t++)
            {
                pool.Schedule([&cd = *this, &r, &rx, &ru, num_term, num_residual,
                               &dim_norm_residual, &weights, &norms, &parameters,
                               &num_norm_parameter, risk, num_sensors,
                               dim_state_derivative, dim_action, dim_max, t, T]()
                              {
                                  // ----- term derivatives ----- //
                                  int f_shift = 0;
                                  int p_shift = 0;
                                  double c = 0.0;
                                  for (int i = 0; i < num_term; i++)
                                  {
                                      c += cd.DerivativeStep(
                                          DataAt(cd.cx, t * dim_state_derivative),
                                          DataAt(cd.cu, t * dim_action),
                                          DataAt(cd.cr, t * num_residual),
                                          DataAt(cd.c_scratch_, t * dim_max * dim_max),
                                          DataAt(cd.cx_scratch_, t * dim_state_derivative),
                                          DataAt(cd.cu_scratch_, t * dim_action),
                                          r + t * num_residual + f_shift,
                                          rx + t * num_sensors * dim_state_derivative +
                                              f_shift * dim_state_derivative,
                                          ru + t * num_sensors * dim_action + f_shift * dim_action,
                                          dim_norm_residual[i], dim_state_derivative, dim_action,
                                          weights[i] / T, parameters + p_shift, norms[i]);

                                      f_shift += dim_norm_residual[i];
                                      p_shift += num_norm_parameter[i];
                                  }

                                  // ----- risk transformation ----- //
                                  if (mju_abs(risk) < kRiskNeutralTolerance)
                                  {
                                      return;
                                  }

                                  double s = mju_exp(risk * c);

                                  // cx
                                  mju_scl(DataAt(cd.cx, t * dim_state_derivative),
                                          DataAt(cd.cx, t * dim_state_derivative), s,
                                          dim_state_derivative);

                                  // cu
                                  mju_scl(DataAt(cd.cu, t * dim_action), DataAt(cd.cu, t * dim_action), s,
                                          dim_action);
                              });
            }
            pool.WaitCount(count_before + T);
        }
        pool.ResetCount();
    }

} // namespace mjpc
