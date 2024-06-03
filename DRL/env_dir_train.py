"""
@Author: Fhz
@Create Date: 2024/3/18 16:28
@File: env_dir_train.py
@Description:
@Modify Person Date:
"""
import os
import random
import sys

# check if SUMO_HOME exists in environment variable
# if not, then need to declare the variable before proceeding
# makes it OS-agnostic

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import gym
from gym import spaces
import numpy as np
import math
import traci
from sumolib import checkBinary
import time
import argparse
import pickle
import pandas as pd



class SumoGym(gym.Env):
    def __init__(self, args):
        self.current_state = None
        self.Acc = None
        self.heading = None
        self.model_name = args.model_name
        self.vehicles = None
        self.prob_main = args.prob_main
        self.prob_merge = args.prob_merge
        self.seed_value = args.seed_value
        self.seed_value1 = args.seed_value1
        self.t_sample = args.t_sample

        # Record log
        self.is_success = False
        self.total_reward = 0
        self.comfort_reward = 0
        self.efficiency_reward = 0
        self.safety_reward = 0
        self.done_count = 0
        # self.gamma = args.gamma

        # SUMO config
        self.count = args.count
        self.show_gui = args.show_gui
        self.sumocfgfile = args.sumocfgfile
        self.egoID = args.egoID
        self.start_time = args.start_time
        self.collision = args.collision
        self.sleep = args.sleep
        self.lane_width = args.lane_width
        self.y_none = args.y_none
        self.vehicle_default_length = args.vehicle_default_length
        self.vehicle_default_width = args.vehicle_default_width
        self.mainRouteProb = args.mainRouteProb
        self.lane_change_time = args.lane_change_time

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
        self.gap = args.gap

        # Reward config
        self.w_jerk_x = args.w_jerk_x
        self.w_jerk_y = args.w_jerk_y
        self.w_time = args.w_time
        self.w_lane = args.w_lane
        self.w_speed = args.w_speed
        self.R_time = args.R_time
        self.V_desired = args.V_desired
        self.R_collision = args.R_collision

        self.P_target = args.P_target
        self.lane_width = args.lane_width

        # Done config
        self.target_lane_id = args.target_lane_id
        self.merge_position = args.merge_position
        self.max_count = args.max_count

        self.trajectory_length = args.trajectory_length
        self.feature_length = args.feature_length

        self.frame = pd.DataFrame(
            columns=["Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "vx", "vy", "Lane_ID"])

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
            self.min_x_acc, # dir

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

        self.action_space = spaces.Box(np.array([-4.5, 0.5], dtype=np.float32) , np.array([2.5, 1.5], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        action_long_new = action[0]
        action_lat = int(action[1])

        if self.is_success:
            self.is_success = False

        if self.collision:
            self.collision = False

        # random.seed(self.seed_value1)


        traci.simulationStep(self.count)

        if self.count % 1 == 0:
            if random.random() < self.prob_main:
                if random.random() < self.mainRouteProb:
                    traci.vehicle.add(vehID="flow_1_{}".format(int(self.count)), routeID="route1", typeID="typedist1",
                                      depart="{}".format(int(self.count)),departLane="0", arrivalLane="{}".format(random.randint(0, 4)))
                else:
                    traci.vehicle.add(vehID="flow_1_{}".format(int(self.count)), routeID="route2", typeID="typedist1",
                                      depart="{}".format(int(self.count)), departLane="0")

                if random.random() < 0.996:
                    traci.vehicle.add(vehID="flow_2_{}".format(int(self.count)), routeID="route1", typeID="typedist1",
                                      depart="{}".format(int(self.count)), departLane="1", arrivalLane="{}".format(random.randint(0, 4)))
                else:
                    traci.vehicle.add(vehID="flow_2_{}".format(int(self.count)), routeID="route2", typeID="typedist1",
                                      depart="{}".format(int(self.count)), departLane="1")

                traci.vehicle.add(vehID="flow_3_{}".format(int(self.count)), routeID="route1", typeID="typedist1",
                                      depart="{}".format(int(self.count)), departLane="2", arrivalLane="{}".format(random.randint(0, 4)))
                traci.vehicle.add(vehID="flow_4_{}".format(int(self.count)), routeID="route1", typeID="typedist1",
                                      depart="{}".format(int(self.count)), departLane="3", arrivalLane="{}".format(random.randint(0, 4)))
                traci.vehicle.add(vehID="flow_5_{}".format(int(self.count)), routeID="route1", typeID="typedist1",
                                      depart="{}".format(int(self.count)), departLane="4", arrivalLane="{}".format(random.randint(0, 4)))

            if random.random() < self.prob_merge:
                traci.vehicle.add(vehID="flow_6_{}".format(int(self.count)), routeID="route3", typeID="typedist1", depart="{}".format(int(self.count)), departLane="0")

        self.historyLoad()
        self.count = self.count + self.t_sample

        ego_lane, left_lane, right_lane = self.getLeftRightLaneId()
        ego_edge = traci.vehicle.getLaneID(self.egoID)

        if ego_edge[:5] == ":gneJ":
            action_lat = 0

        traci.vehicle.setSpeed(self.egoID, max(0, traci.vehicle.getSpeed(self.egoID) + 0.1 * action_long_new))

        # New Speeds
        speeds = max(0, traci.vehicle.getSpeed(self.egoID) + 0.1 * action_long_new)

        if action_lat:
            if (ego_lane - action_lat) >= 0:
                traci.vehicle.changeLane(self.egoID, "{}".format(ego_lane - action_lat), self.lane_change_time)

        self.current_state = self.getVehicleStates()
        Collision_Nums = traci.simulation.getCollidingVehiclesNumber()

        if Collision_Nums:
            print("collision num:{}".format(Collision_Nums))
            self.collision = True

        done = self.getDoneState()
        reward = self.getRewards(action_lat, ego_lane)
        info = {
                "success": self.is_success,
                "total_reward": self.total_reward,
                "comfort_reward": self.comfort_reward,
                "efficiency_reward": self.efficiency_reward,
                "safety_reward": self.safety_reward,
                "collision": self.collision,
                "total_counts": self.count,
                "speeds": speeds,
                }

        if done:
            traci.close()

        return self.current_state, reward, done, info


    def render(self):
        self.show_gui = True
        pass


    def reset(self):
        self.egoID = "self_car"
        self.start_time = 240

        self.collision = False
        if self.show_gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        traci.start([sumoBinary, "-c", self.sumocfgfile])

        if self.sleep:
            time.sleep(2)

        # Reset, Run 24s
        # random.seed(self.seed_value)
        count_success_ego = 0

        for step in range(self.start_time):
            self.count = step * self.t_sample
            traci.simulationStep(self.count)

            # random get new ego vehicle ID
            if self.count >= 10.0 and self.egoID == "self_car":
                IDLists = traci.vehicle.getIDList()
                for veh in IDLists:
                    flow_ID = int(veh[5])
                    depart_time = int(veh[7:])
                    if flow_ID <= 5 and depart_time > 5:
                        route = traci.vehicle.getRouteID(veh)
                        route_id = int(route[-1])
                        if route_id == 2:
                            self.egoID = veh
                            print("The new ego ID is: {}".format(veh))
                            traci.vehicle.setColor(self.egoID, (255, 255, 0))
                            break


            if self.egoID != "self_car":
                count_success_ego += 1
                self.historyLoad()
                if count_success_ego >= 10:
                    break

            if self.count >= 23.0 and self.egoID == "self_car":
                traci.close()
                self.reset()


            if self.count % 1 == 0:
                if random.random() < self.prob_main:
                    if random.random() < self.mainRouteProb:
                        traci.vehicle.add(vehID="flow_1_{}".format(int(self.count)), routeID="route1", typeID="typedist1",
                                          depart="{}".format(int(self.count)), departLane="0", arrivalLane="{}".format(random.randint(0, 4)))
                    else:
                        traci.vehicle.add(vehID="flow_1_{}".format(int(self.count)), routeID="route2", typeID="typedist1",
                                          depart="{}".format(int(self.count)), departLane="0",)

                    if random.random() < 0.996:
                        traci.vehicle.add(vehID="flow_2_{}".format(int(self.count)), routeID="route1", typeID="typedist1",
                                          depart="{}".format(int(self.count)),  departLane="1", arrivalLane="{}".format(random.randint(0, 4)))
                    else:
                        traci.vehicle.add(vehID="flow_2_{}".format(int(self.count)), routeID="route2", typeID="typedist1",
                                          depart="{}".format(int(self.count)), departLane="1",)

                    traci.vehicle.add(vehID="flow_3_{}".format(int(self.count)), routeID="route1", typeID="typedist1",
                                          depart="{}".format(int(self.count)), departLane="2", arrivalLane="{}".format(random.randint(0, 4)))

                    traci.vehicle.add(vehID="flow_4_{}".format(int(self.count)), routeID="route1", typeID="typedist1",
                                          depart="{}".format(int(self.count)), departLane="3", arrivalLane="{}".format(random.randint(0, 4)))

                    traci.vehicle.add(vehID="flow_5_{}".format(int(self.count)), routeID="route1", typeID="typedist1",
                                          depart="{}".format(int(self.count)), departLane="4", arrivalLane="{}".format(random.randint(0, 4)))

                if random.random() < self.prob_merge:

                    traci.vehicle.add(vehID="flow_6_{}".format(int(self.count)), routeID="route3", typeID="typedist1",
                                          depart="{}".format(int(self.count)), departLane="0")


        self.count = self.count + self.t_sample
        self.Acc = traci.vehicle.getAcceleration(self.egoID)
        self.heading = traci.vehicle.getAngle(self.egoID)
        traci.vehicle.setSpeedMode(self.egoID, 0)
        traci.vehicle.setLaneChangeMode(self.egoID, 0)

        return self.getVehicleStates()


    def getRewards(self, action_lat, ego_lane):
        """
        action: action of step
        function: get the reward after action.
        """

        # Comfort reward
        Acc_new = traci.vehicle.getAcceleration(self.egoID)
        heading_new = traci.vehicle.getAngle(self.egoID)

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
        R_speed = -abs(np.sqrt(self.current_state[2] ** 2 + self.current_state[3] **2) - self.V_desired)

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
            min_dist = min(abs(self.current_state[9]), abs(self.current_state[45]))

        R_TTC = -4 / (min_dist + 0.01)
        R_safe = 10 * (R_col + R_TTC)

        R_comfort = max(-100, R_comfort)
        R_eff = max(-30, R_eff)
        R_safe = max(-800, R_safe)

        Rewards = R_comfort + R_eff + R_safe + R_tar

        self.comfort_reward = R_comfort
        self.efficiency_reward = R_eff
        self.safety_reward = R_safe
        self.total_reward = Rewards

        return Rewards

    def getDoneState(self):
        """
        function: get the done state of simulation.
        """
        done = False

        if self.collision:
            done = True
            self.is_success = False
            print("Collision occurs")
            return done

        if self.current_state[0] <= self.P_target and self.current_state[1] >= 300.0:
            self.done_count += 1
            if self.done_count >= 2:
                done = True
                print("Success")
                self.is_success = True
            return done

        if self.count >= self.max_count:
            done = True
            print("Overtime")
            return done

        if self.current_state[1] >= self.merge_position:
            done = True
            print("Exceeding maximum position")
            return done

        return done


    def getVehicleStates(self):
        """
        function: Get all the states of vehicles, observation space.
        """
        # Get the state of ego vehicle
        x_ego, y_ego, x_ego_speed, y_ego_speed, x_ego_acc, y_ego_acc = self.getVehicleStateViaId(self.egoID)
        ego_length = traci.vehicle.getLength(self.egoID)
        ego_width = traci.vehicle.getWidth(self.egoID)

        # get the surrounding vehicle ID
        ego_leader = traci.vehicle.getLeader(self.egoID)
        ego_follower = traci.vehicle.getFollower(self.egoID)
        ego_left_leader = traci.vehicle.getLeftLeaders(self.egoID)
        ego_left_follower = traci.vehicle.getLeftFollowers(self.egoID)
        ego_right_leader = traci.vehicle.getRightLeaders(self.egoID)  # target lane
        ego_right_follower = traci.vehicle.getRightFollowers(self.egoID)  # target lane

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


    def getLeftRightLaneId(self):
        ego_lane = traci.vehicle.getLaneIndex(self.egoID)

        left_ID = ego_lane
        right_ID = ego_lane

        egoLaneName = traci.vehicle.getLaneID(self.egoID)
        edgeID = egoLaneName[:5]

        if self.current_state[1] <=257.79:
            num_ID = 3
        else:
            num_ID = 4

        if ego_lane < num_ID - 1 and ego_lane > 0:
            left_ID = ego_lane + 1
            right_ID = ego_lane - 1
        elif ego_lane == num_ID - 1:
            left_ID = ego_lane
            right_ID = ego_lane - 1
        elif ego_lane == 0:
            left_ID = ego_lane + 1
            right_ID = ego_lane

        if num_ID == 3 and ego_lane == 3:
            left_ID = ego_lane
            right_ID = ego_lane - 1

        if edgeID == "gneE5":
            # print("true")
            ego_lane = 0
            left_ID = ego_lane
            right_ID = ego_lane

        if edgeID == "gneE2":
            # print("true")
            left_ID = max(left_ID, 2)

        if edgeID == "gneE3" or edgeID == ":gneJ":
            # print("true")
            left_ID = 0
            right_ID = 0

        if ego_lane < 0 or ego_lane >=  num_ID:
            ego_lane = 0
            left_ID = ego_lane
            right_ID = ego_lane


        return ego_lane, left_ID, right_ID

    def historyLoad(self):

        vehicleList = traci.vehicle.getIDList()

        frame = self.frame

        vehicle_class = {"passenger": 0, "motorcycle": 1, "truck": 2}

        Local_Y_ego = 0.0
        tra_len = 3.0

        for veh in vehicleList:
            x_ego, y_ego, x_ego_speed, y_ego_speed, x_ego_acc, y_ego_acc = self.getVehicleStateViaId(veh)
            lane_id = traci.vehicle.getLaneIndex(veh)

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

        tra_len = 30

        frame_vehicle = self.frame[self.frame.Vehicle_ID == veh_id]
        x_data = frame_vehicle.values

        if len(x_data) == 0:
            dir = 1.0
        else:
            if len(x_data) < tra_len:
                x_data_new = np.zeros([tra_len, 5])
                for i in range(tra_len):
                    x_data_new[i, :] = x_data[0, 2:] 
                x_data_new[tra_len - len(x_data):, :] = x_data[:, 2:]
            else:
                x_data_new = x_data[-tra_len:, 2:]

            bst = pickle.load(open(self.model_name, "rb"))
            x_test_final = x_data_new.reshape((1, -1))

            preds = bst.predict(x_test_final)
            dir = preds[0]

        return dir


