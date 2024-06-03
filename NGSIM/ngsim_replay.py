"""
@Author: Fhz
@Create Date: 2023/7/25 10:06
@File: ngsim_replay.py
@Description: 
@Modify Person Date: 
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import math
import csv

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary


def get_args():
    parser = argparse.ArgumentParser()
    # SUMO config
    parser.add_argument("--show_gui", type=bool, default=True, help="The flag of show SUMO gui.")
    parser.add_argument("--sumocfgfile", type=str, default="ngsim_config/sumocfg/config_file2.sumocfg",
                        help="The path of the SUMO configure file.")
    parser.add_argument("--dataset", type=str, default="datasets/trajectories-0750am-0805am.csv",
                        help="The path of the dataset.")
    parser.add_argument("--replay_data", type=str, default="data_sumo/data2.csv",
                        help="The path of the replay data.")
    parser.add_argument("--number_replay_vehicles", type=int, default=38, help="The number of replay vehicles.")

    args = parser.parse_args()

    return args


def unitConversion(frame):
    '''
    :param df: data with unit feet
    :return: data with unit meter
    '''
    ft_to_m = 0.3048

    # frame.loc[:, 'Global_Time'] = frame.loc[:, 'Global_Time'] / 100
    for strs in ["Global_X", "Global_Y", "Local_X", "Local_Y", "v_Length", "v_Width", 'v_Vel']:
        frame.loc[:, strs] = frame.loc[:, strs] * ft_to_m

    return frame


def getHeadingAngle(s_state, e_state):
    '''
        :param s_state: start state
        :param e_state: end state
        :return: heading Angle
        '''
    headingAngle = math.atan2((e_state[0] - s_state[0]), (e_state[1] - s_state[1]))
    headingAngle = headingAngle * 180 / math.pi

    return headingAngle


def get_route_ID(lane_start, lane_end, Local_y_start):
    """
    :param lane_start: Lane start
    :param lane_end:  Lane end
    :param Local_y_start:  Lane end
    :return: route ID
    """
    if lane_start <= 5:
        if lane_end <= 5:
            if Local_y_start <= 170:
                route_ID = 1
            else:
                route_ID = 5
        else:
            if Local_y_start <= 170:
                route_ID = 2
            else:
                route_ID = 6
    else:
        if lane_end <= 5:
            if Local_y_start <= 170:
                route_ID = 3
            else:
                route_ID = 5
        else:
            if Local_y_start <= 170:
                route_ID = 4
            else:
                route_ID = 6

    return route_ID


def processOneVehicle(vehicle_ID, t_start, frame):
    frame_new = unitConversion(frame)

    Vehicle_IDs = frame_new.Vehicle_ID.unique()

    f = open('data_sumo/data{}.csv'.format(vehicle_ID), 'w', newline="", encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "v_Class", "v_Vel", "v_Acc", "Lane_ID", "Heading_Angle", "Route_ID"])

    for veh_id in Vehicle_IDs:
        frame_veh = frame_new[frame_new.Vehicle_ID == veh_id]
        if len(frame_veh) > 1:
            frame_veh = frame_veh.values
            for i in range(len(frame_veh) - 1):
                s_state = [frame_veh[i, 5], frame_veh[i, 4]]
                e_state = [frame_veh[i + 1, 5], frame_veh[i + 1, 4]]
                heading = getHeadingAngle(s_state, e_state)

                lane_start = frame_veh[0, 13]
                lane_end = frame_veh[-1, 13]
                Local_y_start = frame_veh[0, 5]
                route_ID = get_route_ID(lane_start, lane_end, Local_y_start)

                csv_writer.writerow([round(veh_id, 1),
                                     0.001 * (frame_veh[i + 1, 3] - t_start),
                                     frame_veh[i + 1, 4],
                                     frame_veh[i + 1, 5],
                                     frame_veh[i + 1, 10],
                                     frame_veh[i + 1, 11],
                                     frame_veh[i + 1, 12],
                                     frame_veh[i + 1, 13],
                                     heading,
                                     route_ID])

    f.close()


def processAllVehicle():
    args = get_args()
    # vehicle_ID = 2

    dataS = pd.read_csv(args.dataset)
    Vehicle_IDs = dataS.Vehicle_ID.unique()

    for vehicle_ID in Vehicle_IDs:
        frame = dataS[dataS.Vehicle_ID == vehicle_ID]
        t_start = np.min(frame.Global_Time.unique())
        t_end = np.max(frame.Global_Time.unique())

        frame_1 = dataS[dataS.Global_Time >= t_start]
        frame_vehicle = frame_1[frame_1.Global_Time <= t_end]

        processOneVehicle(vehicle_ID, t_start, frame_vehicle)
        break


if __name__ == '__main__':
    processAllVehicle()
