"""
@Author: Fhz
@Create Date: 2023/9/4 11:20
@File: gym_ngsim.py
@Description:
@Modify Person Date:
"""
import gym
from gym import Env
from gym import spaces
import numpy as np
from copy import deepcopy
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import pandas as pd
from MTF_LSTM_valid import *

import os
import random
import sys
import traci
from sumolib import checkBinary
import time
import math
import pickle
import xgboost as xgb
import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class gymNGSIM(Env):
    def __init__(self, args, ii):

        self.number = ii

        self.IDLists = args.IDLists
        self.replay_data = args.replay_data
        self.train = args.train

        # Reward config
        self.w_jerk_x = args.w_jerk_x
        self.w_jerk_y = args.w_jerk_y
        self.w_time = args.w_time
        self.w_lane = args.w_lane
        self.w_speed = args.w_speed
        self.R_time = args.R_time
        self.P_lane = args.P_lane
        self.V_desired = args.V_desired
        self.R_collision = args.R_collision
        self.merge_position = args.merge_position
        self.egoStateFlag = True
        self.is_success = False
        self.done_count = 0
        self.egoLane = None
        self.show_gui = args.show_gui
        self.sleep = args.sleep
        self.y_none = args.y_none
        self.vehicle_default_length = args.vehicle_default_length
        self.vehicle_default_width = args.vehicle_default_width
        self.lane_change_time = args.lane_change_time

        self.frame = pd.DataFrame(
            columns=["Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "vx", "vy", "Lane_ID"])

        self.P_za = args.P_za
        self.P_target = args.P_target
        self.P_left = args.P_left
        self.lane_width = args.lane_width

        # Done config
        self.target_lane_id = args.target_lane_id
        self.merge_position = args.merge_position
        self.max_count = args.max_count

        self.start_time = args.start_time

        self.overTime = False
        self.maxPos = False
        self.dataEnd = False


        # self.model = MyLstm(args)
        # self.model.load_state_dict(torch.load(args.model_name))

        self.egoID = None
        self.egoState = None
        self.targetLeaderID = None
        self.targetFollowerID = None
        self.targetLeaderState = None
        self.targetFollowerState = None
        self.dataS1 = None
        self.current_state = None
        self.count = 0
        # self.P_target = args.P_target
        # self.model = MyLstm(args)
        self.model_name = args.model_name
        self.trajectory_length = args.trajectory_length
        self.feature_length = args.feature_length

        # Road config
        self.min_vehicle_length = args.min_vehicle_length
        self.max_vehicle_length = args.max_vehicle_length
        self.min_vehicle_width = args.min_vehicle_width
        self.max_vehicle_width = args.max_vehicle_width
        self.min_x_position = args.min_x_position
        self.max_x_position = args.max_x_position
        self.min_y_position = args.min_y_position
        self.max_y_position = args.max_y_position
        self.min_x_speed = args.min_x_speed
        self.max_x_speed = args.max_x_speed
        self.min_y_speed = args.min_y_speed
        self.max_y_speed = args.max_y_speed
        self.min_x_acc = args.min_x_acc
        self.max_x_acc = args.max_x_acc
        self.min_y_acc = args.min_y_acc
        self.max_y_acc = args.max_y_acc

        self.low = np.array([
            # ego vehicle
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            self.min_vehicle_length,
            self.min_vehicle_width,

            # ego_leader
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            self.min_vehicle_length,
            self.min_vehicle_width,
            self.min_x_acc,  # dir

            # ego_follower
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            self.min_vehicle_length,
            self.min_vehicle_width,
            self.min_x_acc,  # dir

            # ego_left_leader
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            self.min_vehicle_length,
            self.min_vehicle_width,
            self.min_x_acc,  # dir

            # ego_left_follower
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            self.min_vehicle_length,
            self.min_vehicle_width,
            self.min_x_acc,  # dir

            # ego_left_follower
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            self.min_vehicle_length,
            self.min_vehicle_width,
            self.min_x_acc,  # dir

            # ego_right_leader
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            self.min_vehicle_length,
            self.min_vehicle_width,
            self.min_x_acc,  # dir

        ], dtype=np.float32)

        self.high = np.array([
            # ego vehicle
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            self.max_vehicle_length,
            self.max_vehicle_width,

            # ego_leader
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            self.max_vehicle_length,
            self.max_vehicle_width,
            self.max_x_acc,  # dir

            # ego_follower
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            self.max_vehicle_length,
            self.max_vehicle_width,
            self.max_x_acc,  # dir

            # ego_left_leader
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            self.max_vehicle_length,
            self.max_vehicle_width,
            self.max_x_acc,  # dir

            # ego_left_follower
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            self.max_vehicle_length,
            self.max_vehicle_width,
            self.max_x_acc,  # dir

            # ego_left_follower
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            self.max_vehicle_length,
            self.max_vehicle_width,
            self.max_x_acc,  # dir

            # ego_right_leader
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            self.max_vehicle_length,
            self.max_vehicle_width,
            self.max_x_acc,  # dir
        ], dtype=np.float32)

        self.action_space = spaces.Box(np.array([-4.5, 0.5], dtype=np.float32), np.array([2.5, 1.5], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action):
        self.count = round(self.count + 0.1, 1)

        action_long_new = action[0]
        action_lat = int(action[1])

        if self.is_success:
            self.is_success = False

        if self.collision:
            self.collision = False

        self.current_state = self.getVehicleStates()

        if self.count <= self.counts * 0.1 - 0.1:
            # print("The current step is: {}".format(self.count))

            traci.simulationStep(self.count)

            frame_step = self.dataS[self.dataS.Global_Time == self.count]

            vehicle_IDLists = []
            vehicle_IDList = traci.vehicle.getIDList()

            for veh in vehicle_IDList:
                veh_fl = float(veh)
                veh_in = int(veh_fl)
                if veh_in not in self.veh_IDs:
                    traci.vehicle.remove(veh)
                else:
                    if veh_in != self.egoID or self.train:
                        vehicle_IDLists.append(veh)

            if len(frame_step) > 0:
                for i in range(len(vehicle_IDLists)):
                    egoID = float(vehicle_IDLists[i])
                    frame_tmp = frame_step[frame_step.Vehicle_ID == egoID]
                    frame_tmp_value = frame_tmp.values

                    if len(frame_tmp_value) > 0:
                        traci.vehicle.setSpeed(vehID="{}".format(egoID), speed=frame_tmp_value[0, 4])

                        if frame_tmp_value[0, 7] != 1:
                            left_lane, right_lane = self.getLeftAndRightLaneID(frame_tmp_value[0, 6],
                                                                               frame_tmp_value[0, 3])
                            if frame_tmp_value[0, 7] == 0:

                                lane_IDss = traci.vehicle.getLaneID(vehID="{}".format(egoID))
                                # print(lane_IDss[:5])
                                if frame_tmp_value[0, 6] < 7 and lane_IDss[:5] != "gneE3" and lane_IDss[:5] != ":gneJ" and lane_IDss[:5] != "gneE2":
                                    traci.vehicle.changeLane(vehID="{}".format(egoID), laneIndex=left_lane, duration=6)

                            if frame_tmp_value[0, 7] == 2:
                                if frame_tmp_value[0, 6] < 6:
                                    traci.vehicle.changeLane(vehID="{}".format(egoID), laneIndex=right_lane, duration=6)

        else:
            done = True
            print("自车跑完")
            # self.dataEnd = True

        self.historyLoad()

        ego_lane = traci.vehicle.getLaneIndex("{}".format(self.egoID))

        if not self.train:
            ego_edge = traci.vehicle.getLaneID("{}".format(self.egoID))

            if ego_edge[:5] == ":gneJ":
                action_lat = 0

            traci.vehicle.setSpeed("{}".format(self.egoID), max(0, traci.vehicle.getSpeed("{}".format(self.egoID)) + 0.1 * action_long_new))

            if action_lat:
                if (ego_lane - action_lat) >= 0:
                    # print(ego_edge)
                    traci.vehicle.changeLane("{}".format(self.egoID), "{}".format(ego_lane - action_lat), self.lane_change_time)

        speeds = max(0, traci.vehicle.getSpeed("{}".format(self.egoID)) + 0.1 * action_long_new)

        reward = self.getRewardFunction(action_lat, ego_lane)
        done = self.getDoneState()
        info = {
            "success": self.is_success,
            "total_reward": self.total_reward,
            "comfort_reward": self.comfort_reward,
            "efficiency_reward": self.efficiency_reward,
            "safety_reward": self.safety_reward,
            "collision": self.collision,
            "total_counts": self.count,
            "speeds": speeds,
            "overTime": self.overTime,
            "maxPos": self.maxPos,
            # "dataEnd": self.dataEnd,
        }

        if done:
            traci.close()

        return self.current_state, reward, done, info

    def render(self):
        pass

    def reset(self):

        self.count = 0.5
        self.collision = False

        if self.show_gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        # self.replay_data = random.choice(self.IDLists)
        # self.replay_data = "data_sumo_new/data42.csv"
        self.replay_data = self.IDLists[self.number]
        print(self.replay_data)
        self.egoID = float(self.replay_data[27:-4])
        self.sumocfgfile = "ngsim_config/sumocfg/{}/config_file{}.sumocfg".format(self.replay_data[14:22], int(self.egoID))

        traci.start([sumoBinary, "-c", self.sumocfgfile])

        traci.vehicle.setColor("{}".format(self.egoID), (255,0,0))

        dataS = pd.read_csv(self.replay_data)
        self.dataS = self.getReplayFrame(dataS, self.egoID)
        self.frame_ego = self.dataS[self.dataS.Vehicle_ID == self.egoID]

        self.frame_ego['Global_Time'] = self.frame_ego['Global_Time'].round(1)

        self.veh_IDs = self.dataS.Vehicle_ID.unique().tolist()

        for step in range(self.start_time):
            step_new = round((step + 1) * 0.1, 1)
            # print("The current step is: {}".format(step_new))

            traci.simulationStep(step_new)

            frame_step = self.dataS[self.dataS.Global_Time == step_new]

            vehicle_IDLists = []
            vehicle_IDList = traci.vehicle.getIDList()

            for veh in vehicle_IDList:
                veh_fl = float(veh)
                veh_in = int(veh_fl)
                if veh_in not in self.veh_IDs:
                    traci.vehicle.remove(veh)
                else:
                    vehicle_IDLists.append(veh)

            if len(frame_step) > 0:
                for i in range(len(vehicle_IDLists)):
                    egoID = float(vehicle_IDLists[i])
                    frame_tmp = frame_step[frame_step.Vehicle_ID == egoID]
                    frame_tmp_value = frame_tmp.values

                    if len(frame_tmp_value) > 0:
                        traci.vehicle.setSpeed(vehID="{}".format(egoID), speed=frame_tmp_value[0, 4])

                        if frame_tmp_value[0, 7] != 1:
                            left_lane, right_lane = self.getLeftAndRightLaneID(frame_tmp_value[0, 6], frame_tmp_value[0, 3])
                            if frame_tmp_value[0, 7] == 0:

                                lane_IDss = traci.vehicle.getLaneID(vehID="{}".format(egoID))
                                # print(lane_IDss[:5])
                                if frame_tmp_value[0, 6] < 7 and lane_IDss[:5] != "gneE3" and lane_IDss[:5] != ":gneJ":
                                    traci.vehicle.changeLane(vehID="{}".format(egoID), laneIndex=left_lane, duration=6)

                            if frame_tmp_value[0, 7] == 2:
                                if frame_tmp_value[0, 6] < 6:
                                    traci.vehicle.changeLane(vehID="{}".format(egoID), laneIndex=right_lane, duration=6)

        self.counts = len(self.dataS[self.dataS.Vehicle_ID == self.egoID])

        self.historyLoad()
        self.current_state = self.getVehicleStates()

        self.Acc = traci.vehicle.getAcceleration("{}".format(self.egoID))
        self.heading = traci.vehicle.getAngle("{}".format(self.egoID))

        if self.train:
            self.action_space.sample = self.actionSample # Training with replay data, no need in test

        # traci.vehicle.setSpeedMode("{}".format(self.egoID), 0)
        # traci.vehicle.setLaneChangeMode("{}".format(self.egoID), 0)


        self.vehicles = traci.vehicle.getIDList()

        for veh in self.vehicles:
            traci.vehicle.setSpeedMode(veh, 0)
            traci.vehicle.setLaneChangeMode(veh, 0)

        return self.current_state

    def getReplayFrame(self, dataS, egoID):
        frame_tmp = dataS[dataS.Global_Time == 0.0]
        frame_tmp = frame_tmp.sort_values(by='Local_Y')
        frame_tmp_ego = frame_tmp[frame_tmp.Vehicle_ID == egoID]
        frame_7 = frame_tmp[frame_tmp.Lane_ID == 7]  # Lane_ID == 7
        frame_6 = frame_tmp[frame_tmp.Lane_ID == 6]  # Lane_ID == 6
        frame_5 = frame_tmp[frame_tmp.Lane_ID == 5]  # Lane_ID == 5
        frame_4 = frame_tmp[frame_tmp.Lane_ID == 4]  # Lane_ID == 4
        frame_3 = frame_tmp[frame_tmp.Lane_ID == 3]  # Lane_ID == 3
        # frame_2 = frame_tmp[frame_tmp.Lane_ID == 2]  # Lane_ID == 2

        ego_y = frame_tmp_ego.iloc[0, 3]

        # frame_2_front = frame_2[frame_2.Local_Y > ego_y]
        # frame_2_behind = frame_2[frame_2.Local_Y <= ego_y]

        frame_3_front = frame_3[frame_3.Local_Y > ego_y]
        frame_3_behind = frame_3[frame_3.Local_Y <= ego_y]

        frame_4_front = frame_4[frame_4.Local_Y > ego_y]
        frame_4_behind = frame_4[frame_4.Local_Y <= ego_y]

        frame_5_front = frame_5[frame_5.Local_Y > ego_y]
        frame_5_behind = frame_5[frame_5.Local_Y <= ego_y]

        frame_6_front = frame_6[frame_6.Local_Y > ego_y]
        frame_6_behind = frame_6[frame_6.Local_Y <= ego_y]

        frame_7_front = frame_7[frame_7.Local_Y > ego_y]
        frame_7_behind = frame_7[frame_7.Local_Y <= ego_y]

        vehicles = []

        # front
        if len(frame_7_front) > 0:
            vehicles.append(frame_7_front.iloc[0, 0])
        elif len(frame_7_front) > 1:
            vehicles.append(frame_7_front.iloc[1, 0])

        # rear
        if len(frame_7_behind) > 0:
            vehicles.append(frame_7_behind.iloc[-1, 0])
        elif len(frame_7_behind) > 1:
            vehicles.append(frame_7_behind.iloc[-2, 0])

        # front
        if len(frame_6_front) > 0:
            vehicles.append(frame_6_front.iloc[0, 0])
        elif len(frame_6_front) > 1:
            vehicles.append(frame_6_front.iloc[1, 0])

        # rear
        if len(frame_6_behind) > 0:
            vehicles.append(frame_6_behind.iloc[-1, 0])
        elif len(frame_6_behind) > 1:
            vehicles.append(frame_6_behind.iloc[-2, 0])

        # front
        if len(frame_5_front) > 0:
            vehicles.append(frame_5_front.iloc[0, 0])
        elif len(frame_5_front) > 1:
            vehicles.append(frame_5_front.iloc[1, 0])

        # rear
        if len(frame_5_behind) > 0:
            vehicles.append(frame_5_behind.iloc[-1, 0])
        elif len(frame_5_behind) > 1:
            vehicles.append(frame_5_behind.iloc[-2, 0])

        # front
        if len(frame_4_front) > 0:
            vehicles.append(frame_4_front.iloc[0, 0])
        elif len(frame_4_front) > 1:
            vehicles.append(frame_4_front.iloc[1, 0])

        # rear
        if len(frame_4_behind) > 0:
            vehicles.append(frame_4_behind.iloc[-1, 0])
        elif len(frame_4_behind) > 1:
            vehicles.append(frame_4_behind.iloc[-2, 0])

        # front
        if len(frame_3_front) > 0:
            vehicles.append(frame_3_front.iloc[0, 0])
        elif len(frame_3_front) > 1:
            vehicles.append(frame_3_front.iloc[1, 0])

        # rear
        if len(frame_3_behind) > 0:
            vehicles.append(frame_3_behind.iloc[-1, 0])
        elif len(frame_3_behind) > 1:
            vehicles.append(frame_3_behind.iloc[-2, 0])

        # # front
        # if len(frame_2_front) > 0:
        #     vehicles.append(frame_2_front.iloc[0, 0])
        # elif len(frame_2_front) > 1:
        #     vehicles.append(frame_2_front.iloc[1, 0])
        #
        # # rear
        # if len(frame_2_behind) > 0:
        #     vehicles.append(frame_2_behind.iloc[-1, 0])
        # elif len(frame_2_behind) > 1:
        #     vehicles.append(frame_2_behind.iloc[-2, 0])


        frameNew = dataS[dataS.Vehicle_ID == egoID]
        for veh in vehicles:
            frameTmp = dataS[dataS.Vehicle_ID == veh]
            frameNew = pd.concat([frameNew, frameTmp])

        return frameNew


    # def getReplayFrame(self, dataS, egoID):
    #     frame_tmp = dataS[dataS.Global_Time == 0.0]
    #     frame_tmp = frame_tmp.sort_values(by='Local_Y')
    #     frame_tmp_ego = frame_tmp[frame_tmp.Vehicle_ID == egoID]
    #     frame_7 = frame_tmp[frame_tmp.Lane_ID == 7]  # Lane_ID == 7
    #     frame_6 = frame_tmp[frame_tmp.Lane_ID == 6]  # Lane_ID == 6
    #     frame_5 = frame_tmp[frame_tmp.Lane_ID == 5]  # Lane_ID == 5
    #     frame_4 = frame_tmp[frame_tmp.Lane_ID == 4]  # Lane_ID == 4
    #
    #     ego_y = frame_tmp_ego.iloc[0, 3]
    #
    #     frame_4_front = frame_4[frame_4.Local_Y > ego_y]
    #     frame_4_behind = frame_4[frame_4.Local_Y <= ego_y]
    #
    #     frame_5_front = frame_5[frame_5.Local_Y > ego_y]
    #     frame_5_behind = frame_5[frame_5.Local_Y <= ego_y]
    #
    #     frame_6_front = frame_6[frame_6.Local_Y > ego_y]
    #     frame_7_front = frame_7[frame_7.Local_Y > ego_y]
    #
    #     vehicles = []
    #
    #     # vehicles.append(egoID)
    #
    #     # front
    #     if len(frame_7_front) > 0:
    #         vehicles.append(frame_7_front.iloc[0, 0])
    #     elif len(frame_7_front) > 1:
    #         vehicles.append(frame_7_front.iloc[1, 0])
    #
    #     # front
    #     if len(frame_6_front) > 0:
    #         vehicles.append(frame_6_front.iloc[0, 0])
    #     elif len(frame_6_front) > 1:
    #         vehicles.append(frame_6_front.iloc[1, 0])
    #
    #     if len(frame_5_front) > 0:
    #         vehicles.append(frame_5_front.iloc[0, 0])
    #     elif len(frame_5_front) > 1:
    #         vehicles.append(frame_5_front.iloc[1, 0])
    #
    #     # front
    #     if len(frame_4_front) > 0:
    #         vehicles.append(frame_4_front.iloc[0, 0])
    #     elif len(frame_4_front) > 1:
    #         vehicles.append(frame_4_front.iloc[1, 0])
    #
    #     # rear
    #     if len(frame_4_behind) > 0:
    #         vehicles.append(frame_4_behind.iloc[-1, 0])
    #     elif len(frame_4_behind) > 1:
    #         vehicles.append(frame_4_behind.iloc[-2, 0])
    #
    #     if len(frame_5_behind) > 0:
    #         vehicles.append(frame_5_behind.iloc[-1, 0])
    #     elif len(frame_5_behind) > 1:
    #         vehicles.append(frame_5_behind.iloc[-2, 0])
    #
    #     frameNew = dataS[dataS.Vehicle_ID == egoID]
    #     for veh in vehicles:
    #         frameTmp = dataS[dataS.Vehicle_ID == veh]
    #         frameNew = pd.concat([frameNew, frameTmp])
    #
    #     return frameNew


    def getLeftAndRightLaneID(self, Lane_ID, Local_Y):
        if Lane_ID == 1:
            if 412.77 >= Local_Y >= 153.55:  # ego lane index: 5
                left_ID = 5  # lane index
                Right_ID = 4  # lane index
            else:
                left_ID = 4
                Right_ID = 3
        elif Lane_ID == 2:
            if 412.77 >= Local_Y >= 153.55:  # ego lane index: 4
                left_ID = 5  # lane index
                Right_ID = 3  # lane index
            else:
                left_ID = 4
                Right_ID = 2
        elif Lane_ID == 3:
            if 412.77 >= Local_Y >= 153.55:  # ego lane index: 3
                left_ID = 4  # lane index
                Right_ID = 2  # lane index
            else:
                left_ID = 3
                Right_ID = 1
        elif Lane_ID == 4:
            if 412.77 >= Local_Y >= 153.55:  # ego lane index: 2
                left_ID = 3  # lane index
                Right_ID = 1  # lane index
            else:
                left_ID = 2
                Right_ID = 0
        elif Lane_ID == 5:
            if 412.77 >= Local_Y >= 153.55:  # ego lane index: 1
                left_ID = 2  # lane index
                Right_ID = 0  # lane index
            else:
                left_ID = 1
                Right_ID = 0
        elif Lane_ID == 6:
            if 412.77 >= Local_Y >= 153.55:  # ego lane index: 0
                left_ID = 1  # lane index
                Right_ID = 0  # lane index
            else:
                left_ID = 1
                Right_ID = 0
        elif Lane_ID == 7:
            left_ID = 0
            Right_ID = 0
        elif Lane_ID == 8:
            left_ID = 0
            Right_ID = 0
        else:
            print("Error Lane_ID!")

        return left_ID, Right_ID


    def getDoneState(self):

        done = False

        if self.collision:
            done = True
            self.is_success = False
            print("发生碰撞")
            return done

        if self.current_state[0] <= self.P_target and self.current_state[1] >= 300.0:
            self.done_count += 1
            if self.done_count >= 2:
                done = True
                print("成功汇入匝道")
                self.is_success = True
            return done

        if self.count >= self.max_count:
            done = True
            print("超时")
            return done

        if self.current_state[1] >= self.merge_position:
            done = True
            print("超过最大汇入位置")
            return done

        return done

    # def getRewardFunction(self, action_lat, ego_lane):
    #     """
    #             action: action of step
    #             function: get the reward after action.
    #             """
    #
    #     # Comfort reward
    #     Acc_new = traci.vehicle.getAcceleration("{}".format(self.egoID))
    #     heading_new = traci.vehicle.getAngle("{}".format(self.egoID))
    #
    #     Acc_new_x = Acc_new * math.cos(math.pi * heading_new / 180)
    #     Acc_new_y = Acc_new * math.sin(math.pi * heading_new / 180)
    #
    #     Acc_x = self.Acc * math.cos(math.pi * self.heading / 180)
    #     Acc_y = self.Acc * math.sin(math.pi * self.heading / 180)
    #
    #     jerk_x = (Acc_new_x - Acc_x) / 0.1
    #     jerk_y = (Acc_new_y - Acc_y) / 0.1
    #
    #     self.Acc = Acc_new
    #     self.heading = heading_new
    #
    #     R_comfort = - self.w_jerk_x * abs(jerk_x) - self.w_jerk_y * abs(jerk_y)
    #
    #     # Efficient reward
    #     R_time = - self.R_time
    #     R_lane = -abs(self.current_state[0] - self.P_target)
    #     R_speed = -abs(np.sqrt(self.current_state[2] ** 2 + self.current_state[3] ** 2) - self.V_desired)
    #
    #     # if self.is_success:
    #     #     R_tar = 800
    #     # else:
    #     #     R_tar = 0
    #
    #     R_eff = self.w_time * R_time + self.w_lane * R_lane + self.w_speed * R_speed
    #
    #     # Safety Reward
    #     if self.collision:
    #         R_col = self.R_collision
    #     else:
    #         R_col = 0
    #
    #     if action_lat == 0:
    #         min_dist = abs(self.current_state[9])
    #     else:
    #         min_dist = min(abs(self.current_state[9]), abs(self.current_state[41]))
    #
    #     R_TTC = -4 / (min_dist + 0.01)
    #     R_safe = R_col + R_TTC
    #
    #     R_comfort = max(-100, R_comfort)
    #     R_eff = max(-30, R_eff)
    #     R_safe = max(-800, R_safe)
    #
    #     # Rewards = R_comfort + R_eff + R_safe + R_tar
    #     Rewards = R_comfort + R_eff + R_safe
    #
    #     self.comfort_reward = R_comfort
    #     self.efficiency_reward = R_eff
    #     self.safety_reward = R_safe
    #     self.total_reward = Rewards
    #
    #     return Rewards

    def getRewardFunction(self, action_lat, ego_lane):
        """
        action: action of step
        function: get the reward after action.
        """

        # Comfort reward
        Acc_new = traci.vehicle.getAcceleration("{}".format(self.egoID))
        heading_new = traci.vehicle.getAngle("{}".format(self.egoID))

        Acc_new_x = Acc_new * math.cos(math.pi * heading_new / 180)
        Acc_new_y = Acc_new * math.sin(math.pi * heading_new / 180)

        Acc_x = self.Acc * math.cos(math.pi * self.heading / 180)
        Acc_y = self.Acc * math.sin(math.pi * self.heading / 180)

        jerk_x = (Acc_new_x - Acc_x) / 0.1
        jerk_y = (Acc_new_y - Acc_y) / 0.1

        self.Acc = Acc_new
        self.heading = heading_new

        R_comfort = - self.w_jerk_x * abs(jerk_x) - self.w_jerk_y * abs(jerk_y)

        # Efficient reward
        R_time = - self.R_time
        # R_lane = -abs(self.current_state[0] - self.P_target)
        R_speed = -abs(np.sqrt(self.current_state[2] ** 2 + self.current_state[3] ** 2) - self.V_desired)

        if self.is_success:
            R_tar = 100
        else:
            R_tar = 0

        R_eff = 0.05 * (self.w_time * R_time + self.w_speed * R_speed)

        # Safety Reward
        if self.collision:
            R_col = self.R_collision
        else:
            R_col = 0

        if action_lat == 0:
            min_dist = abs(self.current_state[9])
        else:
            min_dist = min(abs(self.current_state[9]), abs(self.current_state[41]))

        R_TTC = -4 / (min_dist + 0.01)
        R_safe = 10 * (R_col + R_TTC)

        R_comfort = max(-100, R_comfort)
        R_eff = max(-30, R_eff)
        R_safe = max(-800, R_safe)

        Rewards = R_comfort + R_eff + R_safe + R_tar
        # Rewards = R_comfort + R_eff + R_safe

        self.comfort_reward = R_comfort
        self.efficiency_reward = R_eff
        self.safety_reward = R_safe
        self.total_reward = Rewards

        return Rewards


    def getVehicleStates(self):
        """
        function: Get all the states of vehicles, observation space.
        """
        x_ego, y_ego, x_ego_speed, y_ego_speed, x_ego_acc, y_ego_acc = self.getVehicleStateViaId(self.egoID)
        ego_length = traci.vehicle.getLength("{}".format(self.egoID))
        ego_width = traci.vehicle.getWidth("{}".format(self.egoID))

        # get the surrounding vehicle ID
        ego_leader = traci.vehicle.getLeader("{}".format(self.egoID))
        ego_follower = traci.vehicle.getFollower("{}".format(self.egoID))
        ego_left_leader = traci.vehicle.getLeftLeaders("{}".format(self.egoID))
        ego_left_follower = traci.vehicle.getLeftFollowers("{}".format(self.egoID))
        ego_right_leader = traci.vehicle.getRightLeaders("{}".format(self.egoID))  # target lane
        ego_right_follower = traci.vehicle.getRightFollowers("{}".format(self.egoID))  # target lane

        # ego leader vehicle
        if ego_leader is None:
            delta_x_leader = 0
            delta_y_leader = self.y_none
            x_leader_speed = x_ego_speed
            y_leader_speed = y_ego_speed
            x_leader_acc = x_ego_acc
            y_leader_acc = y_ego_acc
            leader_length = self.vehicle_default_length
            leader_width = self.vehicle_default_width

            leader_dir = 1
        else:
            leader_id = ego_leader[0]

            x_leader, y_leader, x_leader_speed, y_leader_speed, x_leader_acc, y_leader_acc = self.getVehicleStateViaId(leader_id)
            delta_x_leader = x_leader - x_ego
            delta_y_leader = y_leader - y_ego
            leader_length = traci.vehicle.getLength(leader_id)
            leader_width = traci.vehicle.getWidth(leader_id)
            leader_dir = self.getVehicleIntention(leader_id)

        # ego follower vehicle
        ego_follower_flag = ego_follower[1]
        if ego_follower_flag == -1:
            delta_x_follower = 0
            delta_y_follower = -self.y_none
            x_follower_speed = x_ego_speed
            y_follower_speed = y_ego_speed
            x_follower_acc = x_ego_acc
            y_follower_acc = y_ego_acc
            follower_length = self.vehicle_default_length
            follower_width = self.vehicle_default_width

            follower_dir = 1
        else:
            follower_id = ego_follower[0]

            x_follower, y_follower, x_follower_speed, y_follower_speed, x_follower_acc, y_follower_acc = self.getVehicleStateViaId(
                follower_id)
            delta_x_follower = x_follower - x_ego
            delta_y_follower = y_follower - y_ego
            follower_length = traci.vehicle.getLength(follower_id)
            follower_width = traci.vehicle.getWidth(follower_id)
            follower_dir = self.getVehicleIntention(follower_id)


        # ego left leader vehicle
        if len(ego_left_leader) == 0:
            delta_x_left_leader = self.lane_width
            delta_y_left_leader = self.y_none
            x_left_leader_speed = x_ego_speed
            y_left_leader_speed = y_ego_speed
            x_left_leader_acc = x_ego_acc
            y_left_leader_acc = y_ego_acc
            left_leader_length = self.vehicle_default_length
            left_leader_width = self.vehicle_default_width

            left_leader_dir = 1
        else:
            left_leader_id = ego_left_leader[0][0]

            x_left_leader, y_left_leader, x_left_leader_speed, y_left_leader_speed, x_left_leader_acc, y_left_leader_acc = self.getVehicleStateViaId(
                left_leader_id)
            delta_x_left_leader = x_left_leader - x_ego
            delta_y_left_leader = y_left_leader - y_ego
            left_leader_length = traci.vehicle.getLength(left_leader_id)
            left_leader_width = traci.vehicle.getWidth(left_leader_id)
            left_leader_dir = self.getVehicleIntention(left_leader_id)


        # ego left follower vehicle
        if len(ego_left_follower) == 0:
            delta_x_left_follower = self.lane_width
            delta_y_left_follower = -self.y_none
            x_left_follower_speed = x_ego_speed
            y_left_follower_speed = y_ego_speed
            x_left_follower_acc = x_ego_acc
            y_left_follower_acc = y_ego_acc
            left_follower_length = self.vehicle_default_length
            left_follower_width = self.vehicle_default_width
            left_follower_dir = 1
        else:
            left_follower_id = ego_left_follower[0][0]

            x_left_follower, y_left_follower, x_left_follower_speed, y_left_follower_speed, x_left_follower_acc, y_left_follower_acc = self.getVehicleStateViaId(
                left_follower_id)
            delta_x_left_follower = x_left_follower - x_ego
            delta_y_left_follower = y_left_follower - y_ego
            left_follower_length = traci.vehicle.getLength(left_follower_id)
            left_follower_width = traci.vehicle.getWidth(left_follower_id)
            left_follower_dir = self.getVehicleIntention(left_follower_id)

        # ego right leader vehicle
        if len(ego_right_leader) == 0:
            delta_x_right_leader = -self.lane_width
            delta_y_right_leader = self.y_none
            x_right_leader_speed = x_ego_speed
            y_right_leader_speed = y_ego_speed
            x_right_leader_acc = x_ego_acc
            y_right_leader_acc = y_ego_acc
            right_leader_length = self.vehicle_default_length
            right_leader_width = self.vehicle_default_width
            right_leader_dir = 1
        else:
            right_leader_id = ego_right_leader[0][0]

            x_right_leader, y_right_leader, x_right_leader_speed, y_right_leader_speed, x_right_leader_acc, y_right_leader_acc = self.getVehicleStateViaId(
                right_leader_id)
            delta_x_right_leader = x_right_leader - x_ego
            delta_y_right_leader = y_right_leader - y_ego
            right_leader_length = traci.vehicle.getLength(right_leader_id)
            right_leader_width = traci.vehicle.getWidth(right_leader_id)
            right_leader_dir = self.getVehicleIntention(right_leader_id)

        # ego right follower vehicle
        if len(ego_right_follower) == 0:
            delta_x_right_follower = - self.lane_width
            delta_y_right_follower = - self.y_none
            x_right_follower_speed = x_ego_speed
            y_right_follower_speed = y_ego_speed
            x_right_follower_acc = x_ego_acc
            y_right_follower_acc = y_ego_acc
            right_follower_length = self.vehicle_default_length
            right_follower_width = self.vehicle_default_width

            right_follower_dir = 1
        else:
            right_follower_id = ego_right_follower[0][0]

            x_right_follower, y_right_follower, x_right_follower_speed, y_right_follower_speed, x_right_follower_acc, y_right_follower_acc = self.getVehicleStateViaId(
                right_follower_id)
            delta_x_right_follower = x_right_follower - x_ego
            delta_y_right_follower = y_right_follower - y_ego
            right_follower_length = traci.vehicle.getLength(right_follower_id)
            right_follower_width = traci.vehicle.getWidth(right_follower_id)
            right_follower_dir = self.getVehicleIntention(right_follower_id)


        self.current_state = [
            # ego
            x_ego,
            y_ego,
            x_ego_speed,
            y_ego_speed,
            x_ego_acc,
            y_ego_acc,
            ego_length,
            ego_width,

            # ego leader
            delta_x_leader,
            delta_y_leader,
            x_leader_speed,
            y_leader_speed,
            x_leader_acc,
            y_leader_acc,
            leader_length,
            leader_width,
            leader_dir,

            # ego follower
            delta_x_follower,
            delta_y_follower,
            x_follower_speed,
            y_follower_speed,
            x_follower_acc,
            y_follower_acc,
            follower_length,
            follower_width,
            follower_dir,

            # left leader
            delta_x_left_leader,
            delta_y_left_leader,
            x_left_leader_speed,
            y_left_leader_speed,
            x_left_leader_acc,
            y_left_leader_acc,
            left_leader_length,
            left_leader_width,
            left_leader_dir,

            # left follower
            delta_x_left_follower,
            delta_y_left_follower,
            x_left_follower_speed,
            y_left_follower_speed,
            x_left_follower_acc,
            y_left_follower_acc,
            left_follower_length,
            left_follower_width,
            left_follower_dir,

            # right leader
            delta_x_right_leader,
            delta_y_right_leader,
            x_right_leader_speed,
            y_right_leader_speed,
            x_right_leader_acc,
            y_right_leader_acc,
            right_leader_length,
            right_leader_width,
            right_leader_dir,

            # right follower
            delta_x_right_follower,
            delta_y_right_follower,
            x_right_follower_speed,
            y_right_follower_speed,
            x_right_follower_acc,
            y_right_follower_acc,
            right_follower_length,
            right_follower_width,
            right_follower_dir,
        ]

        return self.current_state


    def getVehicleStateViaId(self, vehicle_ID):
        """
        vehicle_ID: vehicle ID
        function: Get the Vehicle's state via vehicle ID
        """

        vehicle_ID = "{}".format(vehicle_ID)

        # Get the state of ego vehicle
        curr_pos = traci.vehicle.getPosition(vehicle_ID)
        y_ego, x_ego = curr_pos[0], curr_pos[1]
        y_ego_speed = traci.vehicle.getSpeed(vehicle_ID)
        x_ego_speed = traci.vehicle.getLateralSpeed(vehicle_ID)
        acc_ego = traci.vehicle.getAcceleration(vehicle_ID)
        yaw = traci.vehicle.getAngle(vehicle_ID)
        x_ego_acc = acc_ego * math.cos(math.pi * yaw / 180)
        y_ego_acc = acc_ego * math.sin(math.pi * yaw / 180)

        return x_ego, y_ego, x_ego_speed, y_ego_speed, x_ego_acc, y_ego_acc

    def getStates(self, egoID, vehicles):
        arrays = np.zeros(shape=[len(vehicles), 6])
        vehicles_List = []

        x_ego, y_ego, x_ego_speed, y_ego_speed, x_ego_acc, y_ego_acc = self.getVehicleStateViaId(egoID)
        arrays = np.array([x_ego, y_ego, x_ego_speed, y_ego_speed, x_ego_acc, y_ego_acc])
        vehicles_List.append(egoID)

        for i in range(len(vehicles)):
            if vehicles[i] != egoID:
                # print("The vehicle ID: {}".format(vehicles[i]))
                vehicles_List.append(vehicles[i])
                x_tmp, y_tmp, x_tmp_speed, y_tmp_speed, x_tmp_acc, y_tmp_acc = self.getVehicleStateViaId(vehicles[i])
                array_tmp = np.array([x_tmp, y_tmp, x_tmp_speed, y_tmp_speed, x_tmp_acc, y_tmp_acc])
                arrays = np.vstack([arrays, array_tmp])

        if len(vehicles) == 1:
            arrays = arrays.reshape(1, 6)

        return arrays, vehicles_List

    # def getZaFront(self, states, vehicles_List):
    #
    #     ego_leader = traci.vehicle.getLeader("{}".format(self.egoID))
    #
    #     if ego_leader is None:
    #         ego = states[0, :]
    #
    #         x_ind = ego[1] < states[:, 1]
    #         y_ind = (np.abs(self.P_za - states[:, 0])) < self.lane_width
    #         ind = x_ind & y_ind
    #
    #         if ind.sum() > 0:
    #             state_ind = states[ind, :]
    #             zaFront = state_ind[(state_ind[:, 1] - ego[1]).argmin(), :]
    #             index = np.where(states[:, 1] == zaFront[1])
    #             index_new = index[0][0]
    #             # print(vehicles_List)
    #             # print(index_new)
    #             vehicle_id = vehicles_List[int(index_new)]
    #         else:
    #             zaFront = np.asarray([self.P_za, 1000, ego[2], ego[3], ego[4], ego[5]])
    #             vehicle_id = 0
    #     else:
    #         leader_id = ego_leader[0]
    #         x_leader, y_leader, x_leader_speed, y_leader_speed, x_leader_acc, y_leader_acc = self.getVehicleStateViaId(
    #             leader_id)
    #         zaFront = np.asarray([x_leader, y_leader, x_leader_speed, y_leader_speed, x_leader_acc, y_leader_acc])
    #         index = np.where(states[:, 1] == y_leader)
    #         index_new = index[0][0]
    #         vehicle_id = vehicles_List[int(index_new)]
    #         # print(vehicles_List)
    #         # print(index_new)
    #
    #     return zaFront, vehicle_id
    #
    # def getTargetFront(self, states, vehicles_List):
    #     ego = states[0, :]
    #
    #     x_ind = ego[1] < states[:, 1]
    #     y_ind = (np.abs(self.P_target - states[:, 0])) < self.lane_width
    #     ind = x_ind & y_ind
    #
    #     if ind.sum() > 0:
    #         state_ind = states[ind, :]
    #         targetFront = state_ind[(state_ind[:, 1] - ego[1]).argmin(), :]
    #         index = np.where(states[:, 1] == targetFront[1])
    #         index_new = index[0][0]
    #         vehicle_id = vehicles_List[int(index_new)]
    #         # print(vehicles_List)
    #         # print(index_new)
    #     else:
    #         targetFront = np.asarray([self.P_target, 1000, ego[2], ego[3], ego[4], ego[5]])
    #         vehicle_id = 0
    #
    #     return targetFront, vehicle_id
    #
    # def getTargetRear(self, states, vehicles_List):
    #     ego = states[0, :]
    #
    #     x_ind = ego[1] >= states[:, 1]
    #     y_ind = (np.abs(self.P_target - states[:, 0])) < self.lane_width
    #     ind = x_ind & y_ind
    #
    #     if ind.sum() > 0:
    #         state_ind = states[ind, :]
    #         targetRear = state_ind[(state_ind[:, 1] - ego[1]).argmax(), :]
    #         index = np.where(states[:, 1] == targetRear[1])
    #         index_new = index[0][0]
    #         vehicle_id = vehicles_List[int(index_new)]
    #         # print(vehicles_List)
    #         # print(index_new)
    #     else:
    #         targetRear = np.asarray([self.P_target, -1000, ego[2], ego[3], ego[4], ego[5]])
    #         vehicle_id = 0
    #
    #     return targetRear, vehicle_id
    #
    # def getLeftFront(self, states, vehicles_List):
    #     ego = states[0, :]
    #
    #     x_ind = ego[1] < states[:, 1]
    #     y_ind = (np.abs(self.P_left - states[:, 0])) < self.lane_width
    #     ind = x_ind & y_ind
    #
    #     if ind.sum() > 0:
    #         state_ind = states[ind, :]
    #         leftFront = state_ind[(state_ind[:, 1] - ego[1]).argmin(), :]
    #         index = np.where(states[:, 1] == leftFront[1])
    #         index_new = index[0][0]
    #         vehicle_id = vehicles_List[int(index_new)]
    #     else:
    #         leftFront = np.asarray([self.P_left, 1000, ego[2], ego[3], ego[4], ego[5]])
    #         vehicle_id = 0
    #
    #     return leftFront, vehicle_id
    #
    # def getLeftRear(self, states, vehicles_List):
    #     ego = states[0, :]
    #
    #     x_ind = ego[1] >= states[:, 1]
    #     y_ind = (np.abs(self.P_left - states[:, 0])) < self.lane_width
    #     ind = x_ind & y_ind
    #
    #     if ind.sum() > 0:
    #         state_ind = states[ind, :]
    #         leftRear = state_ind[(state_ind[:, 1] - ego[1]).argmax(), :]
    #         index = np.where(states[:, 1] == leftRear[1])
    #         index_new = index[0][0]
    #         vehicle_id = vehicles_List[int(index_new)]
    #     else:
    #         leftRear = np.asarray([self.P_left, -1000, ego[2], ego[3], ego[4], ego[5]])
    #         vehicle_id = 0
    #
    #     return leftRear, vehicle_id



    def getLaneChangLabel(self):

        # 设置换道时长为1s，车道换道点前后0.5s作为换道标签
        lane_change_dic = {}

        frame_ori = self.ego_pd
        t_first = np.min(frame_ori.Global_Time.unique())

        # 初始化赋值为1
        for i in range(len(frame_ori)):
            lane_change_dic.update({"{}".format(round(t_first + i * 0.1, 1)): 1.0})

        for j in range(len(frame_ori) - 1):
            t_tmp = round(t_first + j * 0.1, 1)
            t1 = round(t_tmp + 0.1, 1)
            frame = frame_ori[round(frame_ori.Global_Time, 1) == t_tmp]
            frame_1 = frame_ori[round(frame_ori.Global_Time, 1) == t1]

            lane_id = int(frame.Lane_ID)
            lane_id_1 = int(frame_1.Lane_ID)

            # Store lane change time stamp
            if lane_id > lane_id_1:  # left lane change
                # print("Vehicle ID: {}, time stamp: {}, lane change label: {}".format(i, t_tmp, 0))

                lane_change_dic["{}".format(round(t_tmp - 0.5, 1))] = 0.0
                lane_change_dic["{}".format(round(t_tmp - 0.4, 1))] = 0.0
                lane_change_dic["{}".format(round(t_tmp - 0.3, 1))] = 0.0
                lane_change_dic["{}".format(round(t_tmp - 0.2, 1))] = 0.0
                lane_change_dic["{}".format(round(t_tmp - 0.1, 1))] = 0.0
                lane_change_dic["{}".format(round(t_tmp, 1))] = 0.0
                lane_change_dic["{}".format(round(t_tmp + 0.1, 1))] = 0.0
                lane_change_dic["{}".format(round(t_tmp + 0.2, 1))] = 0.0
                lane_change_dic["{}".format(round(t_tmp + 0.3, 1))] = 0.0
                lane_change_dic["{}".format(round(t_tmp + 0.4, 1))] = 0.0

            elif lane_id < lane_id_1:
                # print("Vehicle ID: {}, time stamp: {}, lane change label: {}".format(i, t_tmp, 2))

                lane_change_dic["{}".format(round(t_tmp - 0.5, 1))] = 2.0
                lane_change_dic["{}".format(round(t_tmp - 0.4, 1))] = 2.0
                lane_change_dic["{}".format(round(t_tmp - 0.3, 1))] = 2.0
                lane_change_dic["{}".format(round(t_tmp - 0.2, 1))] = 2.0
                lane_change_dic["{}".format(round(t_tmp - 0.1, 1))] = 2.0
                lane_change_dic["{}".format(round(t_tmp, 1))] = 2.0
                lane_change_dic["{}".format(round(t_tmp + 0.1, 1))] = 2.0
                lane_change_dic["{}".format(round(t_tmp + 0.2, 1))] = 2.0
                lane_change_dic["{}".format(round(t_tmp + 0.3, 1))] = 2.0
                lane_change_dic["{}".format(round(t_tmp + 0.4, 1))] = 2.0

        return lane_change_dic


    def actionSample(self):

        frame = self.frame_ego[self.frame_ego.Global_Time == round(self.count - 0.1, 1)] # historical
        frame1 = self.frame_ego[self.frame_ego.Global_Time == round(self.count, 1)]

        lane_change_label = frame1.iloc[0, 7]

        if lane_change_label == 1.0:
            lane_change_action = 0.5
            acc_action = (frame1.iloc[0, 3] - frame.iloc[0, 3]) / 0.1
            acc_action = min(acc_action, 2.5)
            acc_action = max(acc_action, -4.5)
        elif lane_change_label == 0.0:
            lane_change_action = 1.5
            acc_action = (frame1.iloc[0, 3] - frame.iloc[0, 3]) / 0.1
            acc_action = min(acc_action, 2.5)
            acc_action = max(acc_action, -4.5)
        else:
            # print("Error action!")
            lane_change_action = 0.5
            acc_action = 0.0

        return np.array([acc_action, lane_change_action])


    def historyLoad(self):

        vehicleList = traci.vehicle.getIDList()

        frame = self.frame

        vehicle_class = {"passenger": 0, "motorcycle": 1, "truck": 2}

        Local_Y_ego = 0.0
        tra_len = 3.0

        for veh in vehicleList:
            x_ego, y_ego, x_ego_speed, y_ego_speed, x_ego_acc, y_ego_acc = self.getVehicleStateViaId(veh)
            lane_id = traci.vehicle.getLaneIndex(veh)
            # Heading = traci.vehicle.getAngle(veh) - 90.0
            # Vtype = traci.vehicle.getVehicleClass(veh)
            # typeValue =  vehicle_class[Vtype]

            time = self.count
            data_tmp = {"Vehicle_ID": veh, "Global_Time": time, "Local_X": x_ego, "Local_Y": y_ego, "vx": x_ego_speed,
                        "vy": y_ego_speed, "Lane_ID": lane_id}

            frame = frame.append(data_tmp, ignore_index=True)
            if veh == self.egoID:
                curr_pos = traci.vehicle.getPosition(self.egoID)
                Local_Y_ego = curr_pos[0]

        frame = frame[frame.Global_Time > self.count - tra_len]
        frame = frame[frame.Local_Y > Local_Y_ego - 100]
        frame = frame[frame.Local_Y < Local_Y_ego + 100]

        self.frame = frame


    def getVehicleIntention(self, veh_id):
        # starttime = datetime.datetime.now()

        tra_len = 30

        frame_vehicle = self.frame[self.frame.Vehicle_ID == veh_id]
        x_data = frame_vehicle.values
        # print(len(x_data))

        if len(x_data) == 0:
            dir = 1.0
        else:
            if len(x_data) < tra_len:
                x_data_new = np.zeros([tra_len, 5])
                for i in range(tra_len):
                    x_data_new[i, :] = x_data[0, 2:] # 首先将矩阵填满所有的最初数据
                x_data_new[tra_len - len(x_data):, :] = x_data[:, 2:] # 然后将x_data填满
            else:
                x_data_new = x_data[-tra_len:, 2:]

            bst = pickle.load(open(self.model_name, "rb"))
            # print((x_data_new))
            x_test_final = x_data_new.reshape((1, -1))
            x_test_final = xgb.DMatrix(x_test_final)

            preds = bst.predict(x_test_final)
            dir = preds[0]

        return dir



class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        print("num_timesteps: {}".format(self.num_timesteps))

        return True


def getOnMergeVehicleID(ii, frame):
    # replay_datas = ["../ngsim_train/processedData/trajectory_2783/data{}.csv",
    #                 "../ngsim_train/processedData/trajectory_1914/data{}.csv",
    #                 "../ngsim_train/processedData/trajectory_1317/data{}.csv", ]

    replay_datas = ["data_sumo_new/data2783/data{}.csv",
                    "data_sumo_new/data1914/data{}.csv",
                    "data_sumo_new/data1317/data{}.csv"]


    frameM = frame[frame.Lane_ID == 7]
    vehicle_ID_List = frameM.Vehicle_ID.unique().tolist()

    List_new = []
    for veh in vehicle_ID_List:
        frameTmp = frame[frame.Vehicle_ID == veh]
        frameTmpValue = frameTmp.values

        # if frameTmpValue[0, 13] == 7:
        if frameTmpValue[0, 13] == 7 and frameTmpValue[-1, 13] < 6:
            List_new.append(replay_datas[ii].format(int(veh)))

    return List_new


def getOffMergeVehicleID(ii, frame):
    # replay_datas = ["../ngsim_train/processedData/trajectory_2783/data{}.csv",
    #                 "../ngsim_train/processedData/trajectory_1914/data{}.csv",
    #                 "../ngsim_train/processedData/trajectory_1317/data{}.csv", ]

    replay_datas = ["data_sumo_new/data2783/data{}.csv",
                    "data_sumo_new/data1914/data{}.csv",
                    "data_sumo_new/data1317/data{}.csv"]


    frameM = frame[frame.Lane_ID == 8]
    vehicle_ID_List = frameM.Vehicle_ID.unique().tolist()

    List_new = []
    for veh in vehicle_ID_List:
        frameTmp = frame[frame.Vehicle_ID == veh]
        frameTmpValue = frameTmp.values

        if frameTmpValue[0, 13] < 6 and frameTmpValue[-1, 13] == 8:
            List_new.append(replay_datas[ii].format(int(veh)))

    return List_new


def get_args():
    parser = argparse.ArgumentParser()
    # SUMO config
    parser.add_argument("--dataset", type=str, default="../ngsim_train/processedData/trajectory_2783_add.csv", help="The path of the dataset.")
    parser.add_argument("--replay_data", type=str, default="../ngsim_train/processedData/trajectory_2783/data{}.csv", help="The path of the replay data.")
    parser.add_argument("--show_gui", type=bool, default=False, help="The flag of show SUMO gui.")
    parser.add_argument("--sleep", type=bool, default=False, help="The flag of sleep of each simulation.")

    # Done config
    parser.add_argument("--target_lane_id", type=int, default=1, help="The ID of target lane.")
    parser.add_argument("--merge_position", type=float, default=410.00, help="The position of the merge lane.")
    parser.add_argument("--max_count", type=int, default=60, help="The maximum length of a training episode.")

    parser.add_argument("--P_left", type=float, default=9.0, help="The lateral position of target lane.")
    parser.add_argument("--P_target", type=float, default=1.8, help="The lateral position of target lane.")
    parser.add_argument("--P_za", type=float, default=1.8, help="The lateral position of za lane.")
    parser.add_argument("--lane_width", type=float, default=3.6, help="The width of a lane.")
    parser.add_argument("--start_time", type=int, default=5, help="The simulation step before learning.")

    # Road config
    parser.add_argument("--min_vehicle_length", type=float, default=0.0, help="The minimum length of vehicle.")
    parser.add_argument("--max_vehicle_length", type=float, default=20.0, help="The maximum length of vehicle.")
    parser.add_argument("--min_vehicle_width", type=float, default=0.0, help="The minimum width of vehicle.")
    parser.add_argument("--max_vehicle_width", type=float, default=8.0, help="The maximum width of vehicle.")
    parser.add_argument("--min_x_position", type=float, default=-40.0, help="The minimum lateral position of vehicle.")
    parser.add_argument("--max_x_position", type=float, default=0.0, help="The maximum lateral position of vehicle.")
    parser.add_argument("--min_y_position", type=float, default=-1.0, help="The minimum longitudinal position of vehicle.")
    parser.add_argument("--max_y_position", type=float, default=1500.0, help="The maximum longitudinal position of vehicle.")
    parser.add_argument("--min_x_speed", type=float, default=-3.0, help="The minimum lateral speed of vehicle.")
    parser.add_argument("--max_x_speed", type=float, default=3.0, help="The maximum lateral speed of vehicle.")
    parser.add_argument("--min_y_speed", type=float, default=0.0, help="The minimum longitudinal speed of vehicle.")
    parser.add_argument("--max_y_speed", type=float, default=40.0, help="The maximum longitudinal speed of vehicle.")
    parser.add_argument("--min_x_acc", type=float, default=-4.5, help="The minimum lateral acceleration of vehicle.")
    parser.add_argument("--max_x_acc", type=float, default=2.5, help="The maximum lateral acceleration of vehicle.")
    parser.add_argument("--min_y_acc", type=float, default=-4.5, help="The minimum longitudinal acceleration of vehicle.")
    parser.add_argument("--max_y_acc", type=float, default=2.5, help="The maximum longitudinal acceleration of vehicle.")

    parser.add_argument("--y_none", type=float, default=2000.0,
                        help="The longitudinal position of a none exist vehicle.")
    parser.add_argument("--vehicle_default_length", type=float, default=5.0, help="The default length of vehicle.")
    parser.add_argument("--vehicle_default_width", type=float, default=2.4, help="The default width of vehicle.")
    parser.add_argument("--lane_change_time", type=float, default=1.0, help="The time of lane change.")


    # Reward config
    parser.add_argument("--w_jerk_x", type=float, default=0.005, help="The weight of lateral jerk reward.")
    parser.add_argument("--w_jerk_y", type=float, default=0.005, help="The weight of longitudinal jerk reward.")
    parser.add_argument("--w_time", type=float, default=0.1, help="The weight of time consuming reward.")
    parser.add_argument("--w_lane", type=float, default=2, help="The weight of target lane reward.")
    parser.add_argument("--w_speed", type=float, default=0.1, help="The weight of desired speed reward.")
    parser.add_argument("--R_time", type=float, default=-0.1, help="The reward of time consuming.")
    parser.add_argument("--P_lane", type=float, default=17.5, help="The lateral position of target lane.")
    parser.add_argument("--V_desired", type=float, default=20.0, help="The desired speed.")
    parser.add_argument("--R_collision", type=float, default=-400, help="The reward of ego vehicle collision.")

    parser.add_argument("--IDLists", type=list, default=[], help="The list of merge in vehicle.")

    parser.add_argument("--model_name", type=str, default="DIR/xgboost.dat", help="The driving intention recognition model.")

    parser.add_argument("--input_size", type=int, default=6, help="The size of input.")
    parser.add_argument("--hidden_size", type=int, default=128, help="The hidden size.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of training process.")
    parser.add_argument("--target_len", type=int, default=10, help="The target trajectory length.")
    parser.add_argument("--teacher_rate", type=float, default=0.5, help="The teacher rate.")

    parser.add_argument("--trajectory_length", type=int, default=30, help="The length of trajectory.")
    parser.add_argument("--feature_length", type=int, default=6, help="The length of feature.")
    parser.add_argument("--train", type=bool, default=False, help="The replay flag of train.")

    args = parser.parse_args()

    return args


def get_returns(arr, gamma):

    returns = 0

    for i in range(len(arr)):
        returns = returns + arr[i] * gamma ** i

    return returns


if __name__ == '__main__':
    args = get_args()
    datasets = [
                "datasets/trajectories-0750am-0805am.csv",
                "datasets/trajectories-0805am-0820am.csv",
                "datasets/trajectories-0820am-0835am.csv",
                ]

    for i in range(len(datasets)):
        dataS = pd.read_csv(datasets[i])
        IDLists_tmp = getOffMergeVehicleID(i, dataS)
        args.IDLists = args.IDLists + IDLists_tmp

    # models = ["model_ngsim/sac_off_tp", "model_ngsim/sac_off_tp1", "model_ngsim/sac_off_tp2"]
    models = ["model_ngsim/sac_off_dir_1", "model_ngsim/sac_off_dir_3", "model_ngsim/sac_off_dir_4"]

    model_results = []
    for model_tmp in models:

        gamma = 0.999
        eposides = len(args.IDLists)
        rewards = []
        speeds = []
        success_count = 0
        collision_count = 0
        counts = 0

        for ii in range(len(args.IDLists)):

            env = gymNGSIM(args, ii)
            model = SAC.load(model_tmp, env=env)

            obs = env.reset()

            done = False
            count = 0
            reward_tmp = []
            speed_tmp = 0
            while not done:
                count += 1
                # time.sleep(0.01)
                # action = env.action_space.sample()
                # print("The action is: {}".format(ACTIONS_ALL[action]))
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                reward_tmp.append(reward)
                speed_tmp = speed_tmp + info["speeds"]

            reward_tmp_ave = get_returns(reward_tmp, gamma)
            speed_tmp_ave = speed_tmp / count

            rewards.append(reward_tmp_ave)
            speeds.append(speed_tmp_ave)
            print(info)

            counts = counts + count
            if info["success"]:
                success_count = success_count + 1

            if info["collision"]:
                collision_count = collision_count + 1

        rewards = np.array(rewards)
        speeds = np.array(speeds)
        # print(
        #     "The rewards is: {}, the robustness is: {}, the speed is: {},  the success_rate is: {},  the collision_rate is: {}, the counts is : {}".format(
        #         rewards.mean(), rewards.std(), speeds.mean(), success_count / eposides, collision_count / eposides,
        #                                                       counts / eposides))
        model_results.append([rewards.mean(),
                              rewards.std(),
                              speeds.mean(),
                              success_count / eposides,
                              collision_count / eposides,
                              0.1 * counts / eposides])

        print("Rewards: {}, Robustness: {}, Speeds: {}, SuccessRate: {}, CollisionRate: {}, Counts: {}".format(
            rewards.mean(),
            rewards.std(),
            speeds.mean(),
            success_count / eposides,
            collision_count / eposides,
            0.1 * counts / eposides))

    np.save("model_results/sac_off_dir_results.npy", arr=np.array(model_results))
