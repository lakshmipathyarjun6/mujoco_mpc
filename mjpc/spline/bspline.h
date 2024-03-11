// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2023
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2022.01.06

#ifndef MJPC_BSPLINE_H_
#define MJPC_BSPLINE_H_

// with modifications - lightweight version of curve fit
// which supports control data replacement

#include "basisfunction.h"

#include <iostream>
#include <numbers>

using namespace std;

namespace mjpc
{
    enum DofType
    {
        DOF_TYPE_ROTATION,
        DOF_TYPE_ROTATION_BALL_X,
        DOF_TYPE_ROTATION_BALL_Y,
        DOF_TYPE_ROTATION_BALL_Z,
        DOF_TYPE_TRANSLATION
    };
    enum MeasurementUnits
    {
        ROT_UNIT_RADIANS,
        ROT_UNIT_DEGREES,
        TRANS_UNIT_METERS,
        TRANS_UNIT_CENTIMETERS,
        TRANS_UNIT_MILLIMETERS
    };

    template <typename Real> class BSplineCurve
    {
    public:
        // Construction.  If the input controls is non-null, a copy is made of
        // the controls.  To defer setting the control points, pass a null
        // pointer and later access the control points via GetControls() or
        // SetControl() member functions.  The domain is t in [t[d],t[n]],
        // where t[d] and t[n] are knots with d the degree and n the number of
        // control points.
        BSplineCurve(int32_t dimension, int32_t degree, int32_t numControls,
                     DofType dofType, MeasurementUnits measurementUnits)
            : mDimension(dimension),
              mControlData(static_cast<size_t>(dimension) *
                           static_cast<size_t>(numControls))
        {
            LogAssert(dimension >= 1,
                      "Invalid dimension (this data is 2D by design).");
            LogAssert(1 <= degree && degree < numControls, "Invalid degree.");

            BasisFunctionInput<Real> input;
            input.numControls = numControls;
            input.degree = degree;
            input.uniform = true;
            input.periodic = false;
            input.numUniqueKnots = numControls - degree + 1;
            input.uniqueKnots.resize(input.numUniqueKnots);
            input.uniqueKnots[0].t = (Real)0;
            input.uniqueKnots[0].multiplicity = degree + 1;
            int32_t last = input.numUniqueKnots - 1;
            Real factor = ((Real)1) / (Real)last;
            for (int32_t i = 1; i < last; ++i)
            {
                input.uniqueKnots[i].t = factor * (Real)i;
                input.uniqueKnots[i].multiplicity = 1;
            }
            input.uniqueKnots[last].t = (Real)1;
            input.uniqueKnots[last].multiplicity = degree + 1;
            mBasis.Create(input);

            mDofType = dofType;
            mMeasurementUnits = measurementUnits;
        }

        // Member access.
        inline BasisFunction<Real> const &GetBasisFunction() const
        {
            return mBasis;
        }

        inline vector<Real> const &GetControlData() const
        {
            return mControlData;
        }

        void SetControlData(vector<Real> &newControlData)
        {
            LogAssert(newControlData.size() == mControlData.size(),
                      "Incompatible control vector dimensions.");

            for (int i = 0; i < mControlData.size(); i++)
            {
                mControlData[i] = newControlData[i];
            }
        }

        // Evaluation of the curve.  The function supports derivative
        // calculation through order 3; that is, order <= 3 is required.  If
        // you want/ only the position, pass in order of 0.  If you want the
        // position and first derivative, pass in order of 1, and so on.  The
        // output array 'jet' must have enough storage to support the maximum
        // order.  The values are ordered as: position, first derivative,
        // second derivative, third derivative.
        void Evaluate(Real t, uint32_t order, Real *value) const
        {
            int32_t imin, imax;
            mBasis.Evaluate(t, order, imin, imax);

            Real const *source =
                &mControlData[static_cast<size_t>(mDimension) * imin];
            Real basisValue = mBasis.GetValue(order, imin);
            for (int32_t j = 0; j < mDimension; ++j)
            {
                value[j] = basisValue * (*source++);
            }

            for (int32_t i = imin + 1; i <= imax; ++i)
            {
                basisValue = mBasis.GetValue(order, i);
                for (int32_t j = 0; j < mDimension; ++j)
                {
                    value[j] += basisValue * (*source++);
                }
            }
        }

        // Compute the (normalized) parametric endpoint of the curve that is
        // impacted by the specified number of control point lookahead. Note
        // that this assumes a uniform knot sequence.
        Real computeParametricEndpoint(double currentParametricTime,
                                       int numKnotPointLookahead)
        {
            int32_t startingKnotIndex = mBasis.GetIndex(currentParametricTime);
            int32_t endingKnotIndex =
                startingKnotIndex + numKnotPointLookahead +
                mBasis.GetDegree(); // correct for knot degree in lookahead

            return mBasis.GetKnotParametricTime(endingKnotIndex);
        }

        void GetPosition(Real t, Real *position) const
        {
            Evaluate(t, 0, position);
        }

        // Get position, but evaluate using the dof type and measurement units
        // Assume dof value is in last dimension of spline curve
        void GetPositionInMeasurementUnits(Real t, Real *position)
        {
            Evaluate(t, 0, position);

            double dofValue = position[mDimension - 1];

            switch (mDofType)
            {
            case DofType::DOF_TYPE_ROTATION:
            case DofType::DOF_TYPE_ROTATION_BALL_X:
            case DofType::DOF_TYPE_ROTATION_BALL_Y:
            case DofType::DOF_TYPE_ROTATION_BALL_Z:
                switch (mMeasurementUnits)
                {
                case MeasurementUnits::ROT_UNIT_RADIANS: // default mujoco units
                    while (dofValue > 2 * numbers::pi)
                    {
                        dofValue -= 2 * numbers::pi;
                    }
                    while (dofValue < -2 * numbers::pi)
                    {
                        dofValue += 2 * numbers::pi;
                    }
                    break;
                case MeasurementUnits::ROT_UNIT_DEGREES:
                    while (dofValue > 360.0)
                    {
                        dofValue -= 360.0;
                    }
                    while (dofValue < -360.0)
                    {
                        dofValue += 360.0;
                    }
                    dofValue *= numbers::pi / 180.0;
                    break;
                default:
                    break;
                }
                break;
            case DofType::DOF_TYPE_TRANSLATION:
                switch (mMeasurementUnits)
                {
                case MeasurementUnits::TRANS_UNIT_METERS: // default mujoco
                                                          // units
                    break;
                case MeasurementUnits::TRANS_UNIT_CENTIMETERS:
                    dofValue *= 0.01;
                    break;
                case MeasurementUnits::TRANS_UNIT_MILLIMETERS:
                    dofValue *= 0.001;
                    break;
                default:
                    break;
                }
                break;
            default:
                cout << "ERROR: Unsupported dof type " << mDofType << endl;
                break;
            }

            position[mDimension - 1] = dofValue;
        }

    private:
        int32_t mDimension;
        BasisFunction<Real> mBasis;
        vector<Real> mControlData;
        DofType mDofType;
        MeasurementUnits mMeasurementUnits;
    };

} // namespace mjpc

#endif // MJPC_BSPLINE_H_
