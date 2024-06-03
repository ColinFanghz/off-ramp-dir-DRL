"""
@Author: Fhz
@Create Date: 2023/7/13 10:14
@File: main_test.py
@Description: 
@Modify Person Date: 
"""
import sys
import os
import time
import numpy as np
import math
import pandas as pd
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

if_show_gui = False

if not if_show_gui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')

sumocfgfile = "../sumo_config/my_config_file1.sumocfg"
traci.start([sumoBinary, "-c", sumocfgfile])

SIM_STEPS = [1, 12000]
beginTime = SIM_STEPS[0]
duration = SIM_STEPS[1]

time.sleep(2)

egoID = "self_car"
prob_main = 0.75
prob_merge = 0.1
t_sample = 0.1

for step in range(duration):
    count = step * t_sample
    traci.simulationStep(count)

    if count % 1 == 0:
        if random.random() < prob_main:
            if random.random() < 0.9:
                traci.vehicle.add(vehID="flow_1_{}".format(count), routeID="route1", typeID="typedist1",
                                  depart="{}".format(count), departLane="0",
                                  arrivalLane="{}".format(random.randint(0, 4)))
            else:
                traci.vehicle.add(vehID="flow_1_{}".format(count), routeID="route2", typeID="typedist1",
                                  depart="{}".format(count), departLane="0")

        if random.random() < prob_main:
            if random.random() < 0.9:
                traci.vehicle.add(vehID="flow_2_{}".format(count), routeID="route1", typeID="typedist1",
                                  depart="{}".format(count), departLane="1",
                                  arrivalLane="{}".format(random.randint(0, 4)))
            else:
                traci.vehicle.add(vehID="flow_2_{}".format(count), routeID="route2", typeID="typedist1",
                                  depart="{}".format(count), departLane="1")

        if random.random() < prob_main:
            if random.random() < 0.9:
                traci.vehicle.add(vehID="flow_3_{}".format(count), routeID="route1", typeID="typedist1",
                                  depart="{}".format(count), departLane="2",
                                  arrivalLane="{}".format(random.randint(0, 4)))
            else:
                traci.vehicle.add(vehID="flow_3_{}".format(count), routeID="route2", typeID="typedist1",
                                  depart="{}".format(count), departLane="2")

        if random.random() < prob_main:
            if random.random() < 0.9:
                traci.vehicle.add(vehID="flow_4_{}".format(count), routeID="route1", typeID="typedist1",
                                  depart="{}".format(count), departLane="3",
                                  arrivalLane="{}".format(random.randint(0, 4)))
            else:
                traci.vehicle.add(vehID="flow_4_{}".format(count), routeID="route2", typeID="typedist1",
                                  depart="{}".format(count), departLane="3")

        if random.random() < prob_main:
            if random.random() < 0.9:
                traci.vehicle.add(vehID="flow_5_{}".format(count), routeID="route1", typeID="typedist1",
                                  depart="{}".format(count), departLane="4",
                                  arrivalLane="{}".format(random.randint(0, 4)))
            else:
                traci.vehicle.add(vehID="flow_5_{}".format(count), routeID="route2", typeID="typedist1",
                                  depart="{}".format(count), departLane="4")

        if random.random() < prob_merge:
            if random.random() < 0.9:
                traci.vehicle.add(vehID="flow_6_{}".format(count), routeID="route3", typeID="typedist1",
                                  depart="{}".format(count), departLane="0",
                                  arrivalLane="{}".format(random.randint(0, 4)))
            else:
                traci.vehicle.add(vehID="flow_6_{}".format(count), routeID="route4", typeID="typedist1",
                                  depart="{}".format(count), departLane="0")
