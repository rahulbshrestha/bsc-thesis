#!/bin/bash

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

# Apply heuristic to it; move solution_analysis to new directory

