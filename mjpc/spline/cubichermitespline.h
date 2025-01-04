#ifndef MJPC_CUBICHERMITESPLINE_H_
#define MJPC_CUBICHERMITESPLINE_H_

using namespace std;

namespace mjpc
{

    template <typename Real> class CubicHermiteSpline
    {
    public:
        // Expected coefficient order:
        // 0: f(0)
        // 1: f'(0)
        // 2: f(1)
        // 3: f'(1)
        // Spline spans domain [0,1]
        CubicHermiteSpline() : mT0Time(0.0), mT1Time(1.0)
        {
            mju_zero(mCoefficients, 4);
        }

        Real *GetCoefficients() { return mCoefficients; }

        Real GetT0Time() { return mT0Time; }

        Real GetT1Time() { return mT1Time; }

        void SetCoefficients(Real *coefficients)
        {
            mju_copy4(mCoefficients, coefficients);
        }

        void SetT0Time(double t0) { mT0Time = t0; }

        void SetT1Time(double t1) { mT1Time = t1; }

        // Source: https://en.wikipedia.org/wiki/Cubic_Hermite_spline
        void Evaluate(Real simTime, Real &value) const
        {
            // Implicitly convert t to be in parametric domain
            double timeDiff = mT1Time - mT0Time;

            double t = 0.0;

            // Prevent divide by 0
            if (timeDiff > 0)
            {
                t = (simTime - mT0Time) / timeDiff;
            }

            double t3 = t * t * t;
            double t2 = t * t;

            double c1 = ((2 * t3) - (3 * t2) + 1) * mCoefficients[0];
            double c2 = (t3 - (2 * t2) + t) * mCoefficients[1];
            double c3 = ((-2 * t3) + (3 * t2)) * mCoefficients[2];
            double c4 = (t3 - t2) * mCoefficients[3];

            value = c1 + c2 + c3 + c4;
        }

    private:
        Real mCoefficients[4];
        Real mT0Time;
        Real mT1Time;
    };

} // namespace mjpc

#endif // MJPC_CUBICHERMITESPLINE_H_