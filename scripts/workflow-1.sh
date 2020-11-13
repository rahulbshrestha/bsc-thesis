#!/bin/bash
# Workflow for applying heuristics to mission solution

# Preparing script 
cd ~/templ-dev/planning/templ/build

if [ "$1" = "-m" ]; then
  make && make install
fi

# Generate missions
./src/templ-mission_generator --config ~/thesis/data/default-mission-generation.xml

# Generate solution from mission
./src/templ-transport_network_planner --mission /tmp/mission.xml --min_solutions 1

# Find the latest directory generated; move solution_analysis to new directory 
SOLUTIONDIR=$(ls /tmp/ | grep templ | sort | tail -1)

cd /tmp/$SOLUTIONDIR
ls
cat solution_analysis.log

# Apply heuristic to it;minimizes delivery time of vehicles.Algorithm 1:vlns(S) [7] : Use VLNS to form a new schedule from previous scheduleS1Sbest←S2T←w−ln0.5cost(S,γ= 0)3fori∈1,....,Ndo4q←random integer in[min(4,|P|),min(100,ξ|P|)]5R←random-items(S,q)6S′←re move solution_analysis to new directory

#cd ~/templ-dev/planning/templ/build
#./src/templ-heuristics --mission /tmp/$SOLUTIONDIR/specs/mission.xml --solution /tmp/$SOLUTIONDIR/0/final_solution_network.gexf --om http://www.rock-robotics.org/2017/11/vrp#
