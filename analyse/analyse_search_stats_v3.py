#/usr/bin/python3
# This script is the updated version -
# after identifying that the solution analyser did not work properly
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
import matplotlib.colors as colors

import tmaz.plot as tikz

parser = argparse.ArgumentParser(description="Generate experiment pictures")
parser.add_argument("-i","--interactive",help="use the script non-interactively",
        action="store_true")
parser.add_argument("-l","--label",help="additional label for the output file")
parser.add_argument("-f","--file",help="solution analysis log file")
args = parser.parse_args()

# Extra the data from the logfile
# and return a hash per identified system with the logdata columns:
# relative time into mission, content-size, localized_pointclouds count,
# request count, spatial constraints
def getData(logfile,
        referenceStartTime = None, referenceEndTime = None):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, logfile)

    logdata = np.loadtxt(filename)
    print(logdata)
    success_cols = logdata[ logdata[:,idxOf('efficacy')] == 1.0 ]
    nosuccess_cols = logdata[ logdata[:,idxOf('efficacy')] != 1.0 ]
    return logdata, success_cols, nosuccess_cols

def idxOf(name):
    array = ['session-id',
              'alpha','beta','sigma',
              'efficacy','efficiency','safety',
              'timehorizon','travel-distance','reconfiguration-cost',
              'total-agent-count','mobile-agent-count'
              #'overall-runtime','solution-runtime','solution-runtime-mean','solution-runtime-stdev',
              #'solution-found','solution-stopped',
              #'propagate','fail','node','depth',
              #'restart','nogood','flaws','cost'
            ]
    return array.index(name)

def createScatterplotMobileAgents(axis, data, title = "", xlabel = "", ylabel = ""):
    legendPatches = []
    axis.yaxis.grid(True)
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_xlim(0.2,1.1)
    axis.set_title(title)

    y = data[:,idxOf("efficiency")],
    x = data[:,idxOf("efficacy")],
    z = data[:,idxOf("safety")],
    agentcount = [ data[:,idxOf("mobile-agent-count")] ]
    cax = axis.scatter(x,y,c = np.array(agentcount), s = 50, marker = 'o',
            cmap="RdYlGn_r",
            alpha=0.8,
            linewidths=0.2,
            edgecolors='grey'
            )
    return cax

def createScatterplotTotalAgents(axis, data, title = "", xlabel = "", ylabel = ""):
    legendPatches = []
    axis.yaxis.grid(True)
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_xlim(0.2,1.1)
    axis.set_title(title)

    y = data[:,idxOf("efficiency")],
    x = data[:,idxOf("efficacy")],
    z = data[:,idxOf("safety")],
    agentcount = [ data[:,idxOf("total-agent-count")] ]
    cax = axis.scatter(x,y,c = np.array(agentcount), s = 50, marker = 'o',
            cmap="RdYlGn_r",
            alpha=0.8,
            linewidths=0.2,
            edgecolors='grey'
            )

    return cax

def createScatterplotSafety(axis, data, best, title = "", xlabel = "", ylabel = ""):
    legendPatches = []
    axis.yaxis.grid(True)
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_xlim(0.1,1.1)
    axis.set_title(title)

    maxPos,maxPerformance, maxEfficiency = best

    y = data[:,idxOf("efficiency")],
    x = data[:,idxOf("efficacy")],
    z = data[:,idxOf("safety")],
    cax = axis.scatter(x,y,c = z, s = 50, marker = 'o',
            cmap="RdYlGn",
            alpha=0.8,
            linewidths=0.2,
            edgecolors='grey',
            vmin = 0.6,
            vmax = 0.9)
    print("Best found: {}".format(best))
    axis.scatter(maxPos[1]+0.03,maxPos[0]*maxEfficiency, c = 'black', s =30, marker = '*', vmin = 0.6, vmax = 0.9)
    return cax

def createParetoLine(axis, data, best, title = "", xlabel = "", ylabel = "",
    xobjective = "efficiency", yobjective = "safety",
    min_efficiency_in_kwh = 0,
    max_efficiency_in_kwh = 6000):
    i = 1.0
    labels = []
    #for i in 0.2,0.4,0.6,0.8,1.0:
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.yaxis.grid(True)
    axis.xaxis.grid(True)
    axis.set_ylim(0.6,0.9)
    axis.set_xlim(min_efficiency_in_kwh-250,max_efficiency_in_kwh + 250)
    tmpdata = data[ data[:, idxOf('efficacy')] == i]
    xo = np.array(tmpdata[:,idxOf( xobjective )])
    yo = np.array(tmpdata[:,idxOf( yobjective )])
    labels.append("{}".format(i))
    cax = axis.plot(xo,yo, 'bo' )
    return cax


def createBoxPlot(axis, data, best, title = "", xlabel = "", ylabel = "",
        objective = "efficiency", ymin = None, ymax = None, final = False):
    legendPatches = []
    axis.yaxis.grid(True)
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    if ymin != None and ymax != None:
        axis.set_ylim(ymin,ymax)

    maxPos,maxPerformance, maxEfficiency = best

    x = []
    y = []
    supportx = []
    supporty = []
    z = []
    labels = []
    for i in 0.2,0.4,0.6,0.8,1.0:
        tmpdata = data[ data[:, idxOf('efficacy')] == i]
        e = np.array(tmpdata[:,idxOf( objective )])
        labels.append("{}".format(i))
        y.append(e)
        supporty.append(len(e))
        z.append(np.std(e))
    cax = axis.boxplot(y)
    axis.set_xticklabels(labels)
    axisright = axis.twinx()
    axisright.plot([1.0,2.0,3.0,4.0,5.0],supporty, color='g',alpha=0.7)
    axisright.tick_params('y',colors='g')
    axisright.set_ylim(0,200)
    axisright.set_ylabel("Number of solutions", color='g')
    if final:
        axisright.get_yaxis().set_visible(True)
    else:
        axisright.get_yaxis().set_visible(False)

    return cax

def maxEfficiency(data, fulfillment = 0.0):
    """
        extract the maximum efficiecy for the given data - which is an array
        of data sets
    """
    maxE = 0
    for i in range(0, len(data)):
        efficacy_mask = data[i][:,idxOf("efficacy")] >= fulfillment
        filtered_data = data[i][efficacy_mask, :]

        e = filtered_data[:,idxOf("efficiency")]
        if np.max(e) > maxE:
            maxE = np.max(e)
    return maxE

def minEfficiency(data, fulfillment = 0.0):
    """
        extract the maximum efficiecy for the given data - which is an array
        of data sets
    """
    minE = sys.maxsize
    for i in range(0, len(data)):
        efficacy_mask = data[i][:,idxOf("efficacy")] >= fulfillment
        filtered_data = data[i][efficacy_mask, :]

        e = filtered_data[:,idxOf("efficiency")]
        if np.min(e) < minE:
            minE = np.min(e)
    return minE

def identifyBest(data, maxEfficiency):
    y = data[:,idxOf("efficiency")],
    x = data[:,idxOf("efficacy")],
    z = data[:,idxOf("safety")],

    ymax = maxEfficiency
    max_performance = 0
    max_pos = []
    # Iterate over all sessions
    for i in range(0,len(y[0])):
        # -1.0, 100, 10
        performance = -y[0][i]/ymax + 100*x[0][i] + 10*z[0][i]
        if performance > max_performance:
            max_performance = performance
            max_pos= [ y[0][i]/ymax,x[0][i],z[0][i],i]

    #mean = [ numpy.mean(y[), numpy.mean(x), numpy.mean(z) ]
    #stdev = [ numpy.stdev(y), numpy.stdev(x), numpy.stdev(z) ]
    #cax = axis.errorplot(base, mean, stdev)
    return max_pos,max_performance, maxEfficiency


if args.label:
    label = args.label

#if args.file:
#    logfile = args.file
#    dataPsi0, successDataPsi0, noSuccessDataPsi0 = getData(logfile)
#else:
#    raise Exception("No log file provided")

dirname = os.path.join( os.path.dirname(__file__),"output")
if not os.path.isdir(dirname):
    os.mkdir(dirname)

solution_dir = "/tmp/20200909_163502+0200-templ"
dataPsi0, successDataPsi0, noSuccessDataPsi0 = getData(f"{solution_dir}/solution_analysis.log")
dataPsi1, successDataPsi1, noSuccessDataPsi1 = getData(f"{solution_dir}/solution_analysis.log")
dataPsi2, successDataPsi2, noSuccessDataPsi2 = getData(f"{solution_dir}/solution_analysis.log")


maxE = maxEfficiency([dataPsi0, dataPsi1, dataPsi2], fulfillment = 0.0)
minE = minEfficiency([dataPsi0, dataPsi1, dataPsi2], fulfillment = 0.0)
print(f"Efficiency range: {minE} -- {maxE}")

bestPsi0 = identifyBest(dataPsi0, maxE)
bestPsi1 = identifyBest(dataPsi1, maxE)
bestPsi2 = identifyBest(dataPsi2, maxE)

print("Best 0: {}".format(bestPsi0))
print("Best 1: {}".format(bestPsi1))
print("Best 2: {}".format(bestPsi2))

maxE1 = maxEfficiency([dataPsi0, dataPsi1, dataPsi2], fulfillment = 1.0)
minE1 = minEfficiency([dataPsi0, dataPsi1, dataPsi2], fulfillment = 1.0)
print(f"Efficiency range (for efficacy 1.0): {minE1} - {maxE1}")

f, axes = plt.subplots(1,3, sharex = True, sharey = True,
        figsize=tikz.Tikz.figsize_by_ratio(0.35))
createParetoLine(axes[0], dataPsi0, bestPsi0, title=r'$\psi_m=0$',
        ylabel = "Safety",
        yobjective = "safety",
        xlabel = "",
        xobjective = "efficiency",
        min_efficiency_in_kwh = minE1,
        max_efficiency_in_kwh = maxE1
        )
createParetoLine(axes[1], dataPsi1, bestPsi1, title=r'$\psi_m=1$',
        ylabel = "",
        yobjective = "safety",
        xlabel = "Efficiency (in kWh)",
        xobjective = "efficiency",
        min_efficiency_in_kwh = minE1,
        max_efficiency_in_kwh = maxE1
        )
createParetoLine(axes[2], dataPsi2, bestPsi2, title=r'$\psi_m=2$',
        ylabel = "",
        yobjective = "safety",
        xlabel = "",
        xobjective = "efficiency",
        min_efficiency_in_kwh = minE1,
        max_efficiency_in_kwh = maxE1
        )
plt.subplots_adjust(right = 0.8)

outfilename = os.path.join(dirname,"templ-psi-comparison-pareto.pdf")
tikz.Tikz.plot(outfilename, { 'interactive': args.interactive, 'type': 'pdf',
    'tight_layout': False, 'align_xlabels': False, 'crop': True})

## display the variance
for objective in "efficiency","safety":
    ymin = None
    ymax = None
    ylabel = ""
    if objective == "efficiency":
        ymin = 0
        ymax = maxE
        ylabel = "Efficiency (in kWh)"
    elif objective == "safety":
        ymin = 0.6
        ymax = 0.9
        ylabel = "Safety"

    f, axes = plt.subplots(1,3, sharex = True, sharey = True, figsize=tikz.Tikz.figsize_by_ratio(0.4))
    createBoxPlot(axes[0], dataPsi0, bestPsi0, title=r'$\psi_m=0$', xlabel = "",
            ylabel = ylabel,
            objective=objective, ymin=ymin, ymax=ymax)
    createBoxPlot(axes[1], dataPsi1, bestPsi1, title=r'$\psi_m=1$', xlabel =
            "Efficacy (as degree of fulfillment)",
            objective=objective, ymin=ymin, ymax=ymax)
    createBoxPlot(axes[2], dataPsi2, bestPsi2, title=r'$\psi_m=2$',
            objective=objective, ymin=ymin, ymax=ymax, final=True)
    plt.subplots_adjust(right = 0.8)

    outfilename = os.path.join(dirname,"templ-psi-comparison-{}.pdf".format(objective))
    tikz.Tikz.plot(outfilename, { 'interactive': args.interactive, 'type': 'pdf',
        'tight_layout': False, 'align_xlabels': False, 'crop': True})

# compare to safety
f, axes = plt.subplots(1,3, sharex = True, sharey = True, figsize=tikz.Tikz.figsize_by_ratio(0.5))
plt.setp(axes, xticks = [0.2,0.4,0.6,0.8,1.0])
scatter = createScatterplotSafety(axes[0], dataPsi0, bestPsi0, title=r'$\psi_m=0$', xlabel = "", ylabel = "Efficiency\n(in kWh)")
createScatterplotSafety(axes[1], dataPsi1, bestPsi1, title=r'$\psi_m=1$', xlabel = "Efficacy\n(as degree of fulfillment)")
scatter = createScatterplotSafety(axes[2], dataPsi2, bestPsi2, title=r'$\psi_m=2$')
plt.subplots_adjust(right = 0.8)
cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
cbar = plt.colorbar(scatter, cax=cax, **kw)
cbar.ax.set_title("Safety")

outfilename = os.path.join(dirname,"templ-psi-comparison-v3.pdf")
tikz.Tikz.plot(outfilename, { 'interactive': args.interactive, 'type': 'pdf',
    'tight_layout': False, 'align_xlabels': False, 'crop': True})

## compare to number of total agents
f, axes = plt.subplots(1,3, sharex = True, sharey = True, figsize=tikz.Tikz.figsize_by_ratio(0.5))
plt.setp(axes, xticks = [0.2,0.4,0.6,0.8,1.0])
createScatterplotTotalAgents(axes[0], dataPsi0, title=r'$\psi_m=0$', xlabel = "", ylabel = "Efficiency\n(in kWh)")
createScatterplotTotalAgents(axes[1], dataPsi1, title=r'$\psi_m=1$', xlabel = "Efficacy\n(as degree of fulfillment)")
scatter = createScatterplotTotalAgents(axes[2], dataPsi2, title=r'$\psi_m=2$')
plt.subplots_adjust(right = 0.8)
cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
cbar = plt.colorbar(scatter, cax=cax, **kw)
cbar.ax.set_title("Atomic Agents")

outfilename = os.path.join(dirname,"templ-psi-comparison-total-agents-v3.pdf")
tikz.Tikz.plot(outfilename, { 'interactive': args.interactive, 'type': 'pdf',
    'tight_layout': False, 'align_xlabels': False, 'crop': True})

# compare to number of mobile agents
f, axes = plt.subplots(1,3, sharex = True, sharey = True, figsize=tikz.Tikz.figsize_by_ratio(0.5))
plt.setp(axes, xticks = [0.2,0.4,0.6,0.8,1.0])
createScatterplotMobileAgents(axes[0], dataPsi0, title=r'$\psi_m=0$', xlabel = "", ylabel = "Efficiency\n(in kWh)")
createScatterplotMobileAgents(axes[1], dataPsi1, title=r'$\psi_m=1$', xlabel = "Efficacy\n(as degree of fulfillment)")
scatter = createScatterplotMobileAgents(axes[2], dataPsi2, title=r'$\psi_m=2$')
plt.subplots_adjust(right = 0.8)
cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
cbar = plt.colorbar(scatter, cax=cax, ticks=[2.0,3.0], **kw)
cbar.ax.set_title("Mobile Agents")
cbar.ax.set_yticklabels(["2","3"])

outfilename = os.path.join(dirname,"templ-psi-comparison-mobile-agents-v3.pdf")
tikz.Tikz.plot(outfilename, { 'interactive': args.interactive, 'type': 'pdf',
    'tight_layout': False, 'align_xlabels': False, 'crop': True})

# identify best solution
