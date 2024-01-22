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

#include <memory>
#include <vector>

#include "mjpc/tasks/acrobot/acrobot.h"
#include "mjpc/tasks/apple/apple.h"
#include "mjpc/tasks/allegro/allegro.h"
#include "mjpc/tasks/hand/hand.h"
#include "mjpc/tasks/humanoid/stand/stand.h"
#include "mjpc/tasks/humanoid/tracking/tracking.h"
#include "mjpc/tasks/humanoid/walk/walk.h"
#include "mjpc/tasks/panda/panda.h"

namespace mjpc
{

  std::vector<std::shared_ptr<Task> > GetTasks()
  {
    return {
        std::make_shared<Acrobot>(),
        std::make_shared<AppleTask>(),
        std::make_shared<AllegroAppleTask>(),
        std::make_shared<AllegroDoorknobTask>(),
        std::make_shared<Hand>(),
        std::make_shared<humanoid::Stand>(),
        std::make_shared<humanoid::Tracking>(),
        std::make_shared<humanoid::Walk>(),
        std::make_shared<Panda>()};
  }
} // namespace mjpc
