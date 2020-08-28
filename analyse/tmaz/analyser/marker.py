from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import argparse
import sys
import time
import datetime as dt
import re
import string
import math
import tmaz.plot as tmaz

class MarkerAnalyser:

    orderedTaskStates = list(reversed([
          #  'PRE_OPERATIONAL',
          #  'STOPPED',
          #  'RUNNING',
            'MOVING_TO_START_POSTURE',
            'START_POSTURE_REACHED',
          #  'START_SEARCHING',
            'SEARCHING',
            'SEARCH_FAILED',
            'MARKER_FOUND',
            'OUTER_MARKER_BOARD',
            'OUTER_ALIGNMENT_REACHED',
            'INNER_MARKER_BOARD',
            'INNER_ALIGNMENT_REACHED',
            'MARKER_DETECTION',
            'ALIGNING',
            'LINEAR_MOVEMENT',
            'LINEAR_MOVEMENT_FINISHED',
            'LIFT_PAYLOAD',
            ]))

    name2taskState = {}

    @staticmethod
    def yticks():
        print("MarkerAnalyser::yticks for %s" % (MarkerAnalyser.orderedTaskStates))
        return np.arange(len(MarkerAnalyser.orderedTaskStates))

    @staticmethod
    def yticklabels():
        yticklabels = []
        for i in MarkerAnalyser.orderedTaskStates:
            label = i.lower()
            label = string.replace(label,"_"," ")
            yticklabels.append(label)
        return yticklabels

    # Extra the data from the logfile
    # and return a hash per identified system with the logdata columns:
    # relative time into mission, content-size, localized_pointclouds count,
    # request count, spatial constraints
    #
    @staticmethod
    def getData(logfile, minTime = 0, maxTime = -1, dataformat = { 'names':
            ('time',
            'state-name', 'state',
            ),
            'formats':
            ( np.float,
              '|S30',np.int
            )
            }, samplingTime = 0, referenceStartTime = None, referenceEndTime = None):
        '''
        Retrieve data form logfile

        :param str logfile: the logfile
        :param int minTime: start of timewindow (relative time)
        :param int maxTime: end of timewindow (relative time), use -1 to get all data
        :param dict dataformat: numpy dataformat description
        '''

        tmaz.Logger.infoGreen("MarkerAnalyser::getData from: {} - min: {}, max: {}".format(logfile, minTime, maxTime))

        logdata = np.loadtxt(logfile,dtype = dataformat)
        # handle the case when there is only one entry -- numpy does not
        # treat that very well it seems
        if logdata.ndim == 0:
            logdata = np.atleast_1d(logdata)


        name2taskState = {}
        data = []

        global_end_time = 0
        global_start_time = -1
        start_time_set = False
        taskState = None
        lastTime = None
        generation_time = None

        # Use absolute start reference time for window definition
        if referenceStartTime != None:
            if isinstance(referenceStartTime, str):
                referenceStartTime = dt.datetime.strptime(referenceStartTime,"%Y%m%d %H:%M:%S")
                referenceStartTime = time.mktime(referenceStartTime.timetuple())
            #elif isinstance(referenceStartTime, float):
            #    # this is what we need
            elif isinstance(referenceStartTime, dt.datetime):
                referenceStartTime = time.mktime(referenceStartTime.timetuple())
                print("Converted datetime: {}".format(referenceStartTime))

            tmaz.Logger.infoGrey("Reference start time set to {}".format(referenceStartTime))

        # Use absolute end reference time for window definition
        if referenceEndTime != None:
            if isinstance(referenceEndTime, str):
                referenceEndTime = dt.datetime.strptime(referenceEndTime,"%Y%m%d %H:%M:%S")
                referenceEndTime = time.mktime(referenceEndTime.timetuple())
            #elif isinstance(referenceEndTime, float):
            #    # this is what we need
            elif isinstance(referenceEndTime, dt.datetime):
                referenceEndTime = time.mktime(referenceEndTime.timetuple())
                print("Converted datetime: {}".format(referenceEndTime))


            tmaz.Logger.infoGrey("Reference end time set to {}".format(referenceEndTime))

        for col in logdata:
            try:
                taskState = MarkerAnalyser.orderedTaskStates.index( col['state-name'] )
                name2taskState[ col['state-name'] ] = taskState
            except ValueError as error:
                # ignore any deactivated tasks
                taskState = None

            try:
                generation_time = col['time']
            except ValueError as error:
                generation_time = None

            if generation_time != None:
                # Compute the relative time to last timestep
                if lastTime != None:
                    time_step = (generation_time - lastTime)

                if lastTime != None and time_step < samplingTime :
                    continue
                else:
                    lastTime = generation_time

                # Check if start time has been set
                # if referenceStartTime is given, this marks the overall
                # start point, otherwise the earliest timestamp in the logfile
                if not start_time_set:
                    if referenceStartTime != None:
                        start_time = referenceStartTime
                    else:
                        start_time = generation_time

                    # Make sure maxTime cannot exceed the selected
                    # end reference time
                    if referenceEndTime != None:
                        maxTime = referenceEndTime - referenceStartTime

                    start_time_set = True

                if global_start_time == -1 or start_time < global_start_time:
                    global_start_time = start_time
                if referenceEndTime != None:
                    global_end_time = referenceEndTime
                elif global_end_time < generation_time:
                    global_end_time = generation_time

                relativeTime = col['time'] - start_time

                if maxTime != -1 and relativeTime > maxTime:
                    break

                if minTime > 0 and relativeTime < minTime:
                    continue

            sample = []
            for field in dataformat['names']:
                if field == 'time':
                    sample.append(relativeTime)
                elif field == 'state':
                    sample.append(taskState)
                else:
                    sample.append(col[field])

            data.append(sample)

        if generation_time != None:
            from_time = dt.datetime.fromtimestamp(global_start_time)
            to_time = dt.datetime.fromtimestamp(global_end_time)
            return data, from_time, to_time, dataformat
        else:
            return data, dataformat

    @staticmethod
    def idxOf(name, dataformat):
        try:
            return dataformat['names'].index(name)
        except ValueError as e:
            print("MarkerAnalyser::idxOf: {} not found in dataformat".format(name))
            raise


    @staticmethod
    def createStateTransitions(axis, data, dataformat,
            location = "Bremen, Germany"):
        columns = np.array(data, dtype=object)
        x = columns[:, MarkerAnalyser.idxOf('time', dataformat)]
        states = columns[:, MarkerAnalyser.idxOf('state', dataformat)]

        state_transitions = []
        current_state = None
        for state in states:
            if current_state == None:
                current_state = state
                state_transitions.append(state)
            elif current_state != state:
                current_state = state
                state_transitions.append(state)
            else:
                state_transitions.append(None)

        if axis != None :
            axis.xaxis.grid(True)
            axis.yaxis.grid(True)
            axis.set_xlabel('time\n(in s)')
            axis.set_ylabel('task state')

            axis.set_yticks(MarkerAnalyser.yticks())
            axis.set_yticklabels(MarkerAnalyser.yticklabels(), {'horizontalalignment': 'left' })
            axis.tick_params("y",pad=130.0)

            axis.plot(x, state_transitions, label="task state activation", marker ="d", markersize = "3", color = 'darkgreen', linestyle="None")
            axis.legend(loc='upper right')

        transitions = []
        for i in range(0,len(x)):
            if state_transitions[i] != None :
                transitions.append({'time': x[i],
                    'state-name': MarkerAnalyser.orderedTaskStates[ state_transitions[i] ]
                    })

        # transition by time and state-name
        return transitions

    @staticmethod
    def identifyTimeOfStateEvent(transitions, statename, previous_occurence = 0,
            startTime = 0, maxTime = -1):
        '''
        Identify the time of an event of entering a task state
        Allow to use additional criteria such as occurence in order to narrow
        the search field
        '''
        count = 0
        for sample in transitions:
            if sample['time'] < startTime:
                continue

            if maxTime > 0 and sample['time'] > maxTime:
                continue

            if statename == sample['state-name']:
                if count == previous_occurence:
                    return sample['time']
                else:
                    count +=1
        raise Exception("No state found corrensponding to name: {}, previous occurence {}, startTime {}".format(statename, previous_occurence, startTime))

    @staticmethod
    def annotateExperiment(axis, from_time, to_time, location, xpos=60, ypos=24):
        tformat = "%H:%M:%S"
        axis.annotate("Robot mission in {} on {}".format(location,
            from_time.strftime("%d %B %Y")), xy=(xpos,ypos), xycoords="figure points")
        axis.annotate("start at: {}".format(from_time.strftime(tformat)),
                xy=(xpos,ypos-12), xycoords="figure points")
        axis.annotate("end at:  {}".format(to_time.strftime(tformat)), xy=(xpos,
            ypos-22), xycoords="figure points")

    @staticmethod
    def createManipulatorMovementXY(axis, data, dataformat):
        columns = np.array(data, dtype=object)
        x = columns[:, MarkerAnalyser.idxOf('target_pose-x', dataformat)]
        y = columns[:, MarkerAnalyser.idxOf('target_pose-y', dataformat)]

        axis.xaxis.grid(True)
        axis.yaxis.grid(True)
        axis.set_xlim(1.0,1.6)
        axis.set_ylim(-0.3,0.3)
        axis.set_xlabel('x-position\n(in m)')
        axis.set_ylabel('y-position\n(in m)')

        axis.plot(x,y, label = "endeffector")
        #axis.legend(loc='upper left')

    @staticmethod
    def createManipulatorMovementZ(axis, data, dataformat):
        columns = np.array(data, dtype=object)
        x = columns[:, MarkerAnalyser.idxOf('time', dataformat)]
        y = columns[:, MarkerAnalyser.idxOf('target_pose-z', dataformat)]

        axis.xaxis.grid(True)
        axis.yaxis.grid(True)
        axis.set_ylim(-0.5,0.5)
        axis.set_xlabel('Time\n(in s)')
        axis.set_ylabel('z-position\n(in m)')
        axis.plot(x,y, label = "endeffector")
        axis.plot(x,y)
        #axis.legend(loc='upper left')

    @staticmethod
    def createManipulatorMovement3D(axis, data, dataformat):
        columns = np.array(data, dtype=object)
        x = columns[:, MarkerAnalyser.idxOf('target_pose-x', dataformat)]
        y = columns[:, MarkerAnalyser.idxOf('target_pose-y', dataformat)]
        z = columns[:, MarkerAnalyser.idxOf('target_pose-z', dataformat)]

        axis.set_xlim(0,2.0)
        axis.set_ylim(-0.5,0.5)
        axis.set_zlim(-0.5,0.5)
        axis.set_xlabel("x position\n(in m)")
        axis.set_ylabel("y position\n(in m)")
        axis.set_zlabel("z position\n(in m)")
        axis.scatter3D(x,y,z, c=z, cmap='Greens')

    @staticmethod
    def createAlignmentComparisonBoxPlot(axis, data, dataformat):
        columns = np.array(data, dtype=object)
        xticks = []
        duration_data = []
        for action in "pick-payload","place-payload":
            # only action with success
            action_columns = columns[ columns[:,MarkerAnalyser.idxOf("action", dataformat)] == action ]
            action_columns = action_columns[ action_columns[:, MarkerAnalyser.idxOf("success", dataformat)] == 'True' ]
            print("Columns {}\n{}\n".format(action,columns))
            for label in "inner","outer":
                alignment_duration = action_columns[:, MarkerAnalyser.idxOf("{}_alignment_duration".format(label), dataformat)]
                xticks.append("{}\n({})".format(action, label))
                duration_data.append(alignment_duration)

        bplot = axis.boxplot(duration_data, patch_artist=True)

        colors = { 'inner': 'lightblue', 'outer': 'lightgreen' }
        bplot['boxes'][0].set_facecolor(colors['inner'])
        bplot['boxes'][1].set_facecolor(colors['outer'])

        bplot['boxes'][2].set_facecolor(colors['inner'])
        bplot['boxes'][3].set_facecolor(colors['outer'])

        axis.set_ylabel("Time\n(in seconds)")
        axis.set_xticks([ 1.5, 3.5])
        axis.set_xticklabels(["pick payload", "place payload"])

        inner_patch = mpatches.Patch(color=colors['inner'], label = "inner alignment")
        outer_patch = mpatches.Patch(color=colors['outer'], label = "outer alignment")
        axis.legend(handles=[inner_patch,outer_patch], loc='upper left')

    @staticmethod
    def createAlignmentComparison(axis, data, dataformat, label = "outer",
            action = "mixed"):
        columns = np.array(data, dtype=object)
        if action != "mixed":
            columns = columns[ columns[:,0] == action ]

        x = columns[:, MarkerAnalyser.idxOf('target_pose-x', dataformat)]
        y = columns[:, MarkerAnalyser.idxOf('target_pose-y', dataformat)]
        yaw = columns[:, MarkerAnalyser.idxOf('target_pose-yaw', dataformat)]
        alignment_duration = columns[:, MarkerAnalyser.idxOf("{}_alignment_duration".format(label), dataformat)]

        success = columns[:, MarkerAnalyser.idxOf('success', dataformat)]
        success_x = []
        success_y = []

        failure_x = []
        failure_y = []
        idx = 0
        for s in success:
            if s == 'True':
                success_x.append(x[idx])
                success_y.append(y[idx])
            else:
                failure_x.append(x[idx])
                failure_y.append(y[idx])
            idx += 1

        axis.set_xlim(1.0,1.7)
        axis.set_ylim(-0.35,0.35)
        axis.set_xlabel("x position\n(in m)")
        axis.set_ylabel("y position\n(in m)")
        axis.xaxis.grid(True)
        axis.yaxis.grid(True)
        axis.set_axisbelow(True)

        min_duration = None
        max_duration = None
        for i in alignment_duration:
            # Skip when search failed
            if i == 0:
                continue

            if min_duration == None or i < min_duration:
                min_duration = i

            if max_duration == None or i > max_duration:
                max_duration = i

        print("Min/max duration {} {}".format(min_duration, max_duration))

        x_new = []
        y_new = []
        alignment_duration_new = []
        for i in range(0,len(x)):
            # diamond marker
            marker_style = mpl.markers.MarkerStyle('|')
            # -90 to account for (diamond being upside down in 0 state, while we
            # want to indicate long side corresponding to robot's (coyote iii)
            # long side
            marker_style._transform = marker_style.get_transform().rotate_deg(yaw[i]*180/math.pi - 90)
            if(alignment_duration[i] != 0):
                axis.scatter( (x[i]) , (y[i]) , s = 250, marker=marker_style,
                        c = alignment_duration[i], vmin = min_duration,
                        vmax = max_duration)
                x_new.append( x[i] )
                y_new.append( y[i] )
                alignment_duration_new.append( alignment_duration[i] )
            else:
                axis.plot( (x[i]) , (y[i]) , markersize = 10, marker="*", color = 'pink' )

        cax = axis.scatter(x_new,y_new, c = alignment_duration_new, marker='o')

        f = plt.gcf()
        cbar = f.colorbar(cax,ticks=[ min(alignment_duration_new),
            np.average(alignment_duration_new),
            max(alignment_duration_new)] )
        cbar.ax.set_yticklabels(["min: {:2.1f} s".format(min(alignment_duration_new)),
            "mean: {:2.1f} s".format(np.average(alignment_duration_new)),
            "max: {:2.1f} s".format(max(alignment_duration_new))])

        cbar.ax.set_title("Time", fontsize=10)
        if label != "outer":
            axis.scatter(failure_x,failure_y, s = 5, marker = 'x', color ='red')

        if action == "mixed":
            axis.set_title("{} alignment (pickup and place)".format(label))
        else:
            axis.set_title("{} alignment ({})".format(label,action))


        patches = []
        payload_patch = Line2D([0],[0],marker='o', color='blue', label='payload pose',
                markerfacecolor='blue', markersize = 5)
        patches.append(payload_patch)

        if len(alignment_duration_new) != len(alignment_duration):
            search_patch = Line2D([0],[0],marker='*', color='white', label='failed search',
                    markerfacecolor='pink', markersize = 10)
            patches.append(search_patch)

        if label != "outer":
            red_patch = Line2D([0],[0],marker='o', color='white', label='failed action',
                    markerfacecolor='red', markersize = 5)
            patches.append(red_patch)

        if len(patches) > 0:
            axis.legend(handles = patches, loc='upper right')
        # Account only for search failure for outer alignment
        if label == "outer":
            positive = len(alignment_duration_new)
            negative = len(alignment_duration) - positive
        else:
            positive = len(success_x)
            negative = len(failure_y)

        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha = 0.9)
        axis.annotate("success: {}, failed: {}\nsuccess rate: {:0.2f}".format(positive,
            negative, positive*1.0/((positive+negative)*1.0)), xy=(1.02,0.235),
            xycoords="data",
            bbox=bbox_props)


    @staticmethod
    def getColsAt(time, data, dataformat, cols):
        '''
        Get all col entry given by col for a specific time point
        Note, that if the timepoint cannot be exactly matched, then the previous
        closest will be used
        '''
        lastsample = None
        sampletime = None
        selectedsample = None
        for sample in data:
            sampletime = sample[ MarkerAnalyser.idxOf('time', dataformat) ]
            if time == sampletime:
                selectedsample = sample
                break

            if sampletime >= time:
                print("MarkerAnalyser::getColsAt: could not find exact match for time {} so using time {}".format(time, sampletime))
                selectedsample = lastsample
                break

            lastsample = sample

        success = True
        if sampletime == None:
            raise Exception("MarkerAnalyser::getColsAt: could not extract sample time from data {}".format(data))

        distance_in_seconds = 2
        if abs(sampletime - time) <= distance_in_seconds:
            selectedsample = lastsample
        else:
            print("MarkerAnalyser::getColsAt: no sample found for time within {} seconds distance {} -- using last sample at {}".format(distance_in_seconds, time, sampletime))
            success = False
            selectedsample = lastsample

        if selectedsample != None:
            coldata = {}
            for colname in cols:
                coldata[colname] = selectedsample[ MarkerAnalyser.idxOf(colname,dataformat) ]
            return coldata, success


    @staticmethod
    def createForceProfile(axis, data, dataformat):
        columns = np.array(data, dtype=object)
        x = columns[:, MarkerAnalyser.idxOf('time', dataformat)]
        forceX = columns[:,MarkerAnalyser.idxOf('force-x', dataformat)]
        forceY = columns[:,MarkerAnalyser.idxOf('force-y', dataformat)]
        forceZ = columns[:,MarkerAnalyser.idxOf('force-z', dataformat)]

        axis.xaxis.grid(True)
        axis.yaxis.grid(True)
        axis.set_ylabel("Force\n(in Nm)")
        axis.set_xlabel("Time\n(in s)")
        axis.plot(x, forceX, label= "x")
        axis.plot(x, forceY, label= "y")
        axis.plot(x, forceZ, label= "z")
        axis.legend(loc='lower right')

    @staticmethod
    def createTypeData(axis, data, label, idxName, robot2color, dataformat, barwidth = 0.2):
        '''
        Print the sample count for a given column over time
        '''
        axis.xaxis.grid(True)
        axis.yaxis.grid(True)
        #axis.set_ylim(0,45)
        axis.set_xlabel('Time\n(in seconds)')
        axis.set_ylabel('{} samples\n(count)'.format(label))
        axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        for robot in data.keys():
            columns = np.array(data[robot])
            x = columns[:,MarkerAnalyser.idxOf('time', dataformat)]
            y = columns[:,MarkerAnalyser.idxOf(idxName, dataformat)]

            # Select Current Axis (SCA) before calling xticks
            plt.sca(axis)
            axis.bar(x, y, barwidth, color = robot2color[robot])



    @staticmethod
    def createRequestTypeData(axis, data, dataformat):
        '''
            :param dictionary data: robot to samples
        '''
        for robot in data.keys():
            columns = np.array(data[robot])
            # Time
            x = columns[:,MarkerAnalyser.idxOf('time', dataformat)]
            # localized pointclouds
            localized_pointclouds = columns[:,MarkerAnalyser.idxOf('localized_pointclouds',dataformat)]
            # requests
            requests = columns[:, MarkerAnalyser.idxOf('requests',dataformat)]
            # spatial_constraints
            spatial_constraints = columns[:, MarkerAnalyser.idxOf('spatial_constraints',dataformat)]
            # footprints
            footprints = columns[:, MarkerAnalyser.idxOf('footprints', dataformat)]
            # action msg
            action_msgs = columns[:, MarkerAnalyser.idxOf('action_msgs', dataformat)]

            axis.xaxis.grid(True)
            axis.yaxis.grid(True)
            axis.set_ylim(0,10)
            axis.set_xlabel('Time\n(in seconds)')
            axis.set_ylabel('distributed mapping samples\n(count)')
            axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

            # Select Current Axis (SCA) before calling xticks
            plt.sca(axis)

            colors = [ "darkmagenta", "steelblue","y" ]
            axis.bar(x,localized_pointclouds, 2, color= colors[0])
            bottom_height = localized_pointclouds

            axis.bar(x,requests, 2, color=colors[1], bottom = bottom_height)
            bottom_height += requests

            axis.bar(x, spatial_constraints, 2, color=colors[2], bottom = bottom_height)
            bottom_height += spatial_constraints

            #axis.bar(x, action_msgs, 2, color="orange", bottom = bottom_height)
            #bottom_height += action_msgs

        #gray_patch = mpatches.Patch(color="grey", label = "footprints")
        red_patch = mpatches.Patch(color=colors[0], label = "localized pointcloud")
        green_patch = mpatches.Patch(color=colors[1], label = "requests")
        purple_patch = mpatches.Patch(color=colors[2], label = "spatial constraints")
        #orange_patch = mpatches.Patch(color="orange", label = "action msg")

        axis.legend(handles=[red_patch, green_patch, purple_patch], loc='upper left')
