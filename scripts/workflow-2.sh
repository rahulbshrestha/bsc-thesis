#!/bin/bash
# Workflow for changing constraints before planning

# Preparing script 
cd ~/templ-dev/planning/templ/build

if [ "$1" = "-m" ]; then
  make && make install
fi

# Generate solution from mission with default config, runs for set time amount

#./src/templ-transport_network_planner --mission ~/templ-dev/planning/templ/test/data/scenarios/should_succeed/0.xml --min_solutions 1 --configuration ~/templ-dev/planning/templ/test/data/configuration/default-configuration.xml
timeout 15m ./src/templ-transport_network_planner --mission ~/thesis/data/missions/mission-1/mission-1.xml --min_solutions 75 --configuration ~/thesis/data/missions/configuration.xml

# Find the latest directory generated; move solution_analysis to new directory 
SOLUTIONDIR=$(ls /tmp/ | grep templ | sort | tail -1)

cd /tmp/$SOLUTIONDIR
ls
cat solution_analysis.log

