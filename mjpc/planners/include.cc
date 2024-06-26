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

#include "mjpc/planners/include.h"

#include <memory>
#include <vector>

#include "mjpc/planners/planner.h"

#include "mjpc/planners/bsplinepd/planner.h"
#include "mjpc/planners/bsplinesampling/planner.h"
#include "mjpc/planners/nothing/planner.h"
#include "mjpc/planners/pcbsplinemanual/planner.h"
#include "mjpc/planners/pcbsplinepd/planner.h"
#include "mjpc/planners/pcbsplinesampling/planner.h"

using namespace std;

namespace mjpc
{
    const char kPlannerNames[] = "Nothing\n"
                                 "BSplinePD\n"
                                 "BSplineSampling\n"
                                 "PCBSplinePD\n"
                                 "PCBSplineManual\n"
                                 "PCBSplineSampling\n";

    // load all available planners
    vector<unique_ptr<mjpc::Planner>> LoadPlanners()
    {
        // planners
        vector<unique_ptr<mjpc::Planner>> planners;

        planners.emplace_back(new mjpc::NothingPlanner);
        planners.emplace_back(new mjpc::BSplinePDPlanner);
        planners.emplace_back(new mjpc::BSplineSamplingPlanner);
        planners.emplace_back(new mjpc::PCBSplinePDPlanner);
        planners.emplace_back(new mjpc::PCBSplineManualPlanner);
        planners.emplace_back(new mjpc::PCBSplineSamplingPlanner);

        return planners;
    }

} // namespace mjpc
