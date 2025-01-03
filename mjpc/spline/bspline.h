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

#define DEFAULT_FRAMERATE 120.0

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
                     DofType dofType, MeasurementUnits measurementUnits,
                     Real frameRate =
                         DEFAULT_FRAMERATE) // TODO: Remove default framerate
            : mDimension(dimension), mFramerate(frameRate),
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

            int index = mDimension * imin;

            Real basisValue = mBasis.GetValue(order, imin);
            for (int32_t j = 0; j < mDimension; ++j)
            {
                value[j] = basisValue * mControlData[index];
                index++;
            }

            for (int32_t i = imin + 1; i <= imax; ++i)
            {
                basisValue = mBasis.GetValue(order, i);
                for (int32_t j = 0; j < mDimension; ++j)
                {
                    value[j] += basisValue * mControlData[index];
                    index++;
                }
            }
        }

        void GetContributingControlPointRangeForTime(
            Real t, int32_t &startControlIndex, int32_t &endControlIndex) const
        {
            int32_t tKnotIndex = mBasis.GetIndex(t);
            int degree = mBasis.GetDegree();

            startControlIndex = tKnotIndex - degree;
            endControlIndex = tKnotIndex;
        }

        // Get position and velocity, but evaluate using the dof type and
        // measurement units Assume dof value is in last dimension of spline
        // curve
        void GetPositionAndVelocityInMeasurementUnits(Real t, Real &position,
                                                      Real &velocity)
        {
            vector<double> posBuff;
            posBuff.resize(mDimension);

            vector<double> velBuff;
            velBuff.resize(mDimension);

            Evaluate(t, 0, posBuff.data());
            Evaluate(t, 1, velBuff.data());

            double posValue = posBuff[mDimension - 1];
            double velValue = velBuff[mDimension - 1];

            double frameTimesteps = velBuff[0];

            velValue /= frameTimesteps;
            velValue *= mFramerate;

            switch (mDofType)
            {
            case DofType::DOF_TYPE_ROTATION:
            case DofType::DOF_TYPE_ROTATION_BALL_X:
            case DofType::DOF_TYPE_ROTATION_BALL_Y:
            case DofType::DOF_TYPE_ROTATION_BALL_Z:
                switch (mMeasurementUnits)
                {
                case MeasurementUnits::ROT_UNIT_RADIANS: // default mujoco units
                    while (posValue > 2 * numbers::pi)
                    {
                        posValue -= 2 * numbers::pi;
                    }
                    while (posValue < -2 * numbers::pi)
                    {
                        posValue += 2 * numbers::pi;
                    }
                    break;
                case MeasurementUnits::ROT_UNIT_DEGREES:
                    while (posValue > 360.0)
                    {
                        posValue -= 360.0;
                    }
                    while (posValue < -360.0)
                    {
                        posValue += 360.0;
                    }
                    posValue *= numbers::pi / 180.0;
                    velValue *= numbers::pi / 180.0;
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
                    posValue *= 0.01;
                    velValue *= 0.01;
                    break;
                case MeasurementUnits::TRANS_UNIT_MILLIMETERS:
                    posValue *= 0.001;
                    velValue *= 0.001;
                    break;
                default:
                    break;
                }
                break;
            default:
                cout << "ERROR: Unsupported dof type " << mDofType << endl;
                break;
            }

            position = posValue;
            velocity = velValue;
        }

    private:
        int32_t mDimension;
        Real mFramerate;
        BasisFunction<Real> mBasis;
        vector<Real> mControlData;
        DofType mDofType;
        MeasurementUnits mMeasurementUnits;
    };

} // namespace mjpc

#endif // MJPC_BSPLINE_H_
