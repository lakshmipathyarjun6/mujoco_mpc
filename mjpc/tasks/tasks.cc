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

#include "mjpc/tasks/tasks.h"

#include "mjpc/tasks/MANO/MANO.h"
#include "mjpc/tasks/allegro/allegro.h"

namespace mjpc
{

    vector<shared_ptr<Task>> GetTasks()
    {
        return {make_shared<AllegroAppleTask>(),
                make_shared<AllegroDoorknobTask>(),
                make_shared<MANOAppleTask>()};
        // make_shared<MANODoorknobTask>()};
    }

} // namespace mjpc
