"""
@Author: Fhz
@Create Date: 2022/11/6 20:51
@File: final_DP.py
@Description: 
@Modify Person Date: 
"""
import pandas as pd
import numpy as np
import time


def getTargetVehicle(path):
    dataS = pd.read_csv(path)
    veh_list = dataS.Vehicle_ID.unique()
    veh_left = []
    veh_center = []
    veh_right = []

    for veh_id in veh_list:
        frame_ori = dataS[dataS.Vehicle_ID == veh_id]
        LC_list = frame_ori.Lane_Change_Label.unique()
        for LC_id in LC_list:
            if LC_id == 0:
                veh_left.append(veh_id)
            elif LC_id == 1:
                veh_center.append(veh_id)
            else:
                veh_right.append(veh_id)

    veh_new = list(set(veh_left + veh_right))
    veh_new_np = np.array(veh_new)
    veh_new_np.sort()

    return veh_new_np


class featureExtract():
    def __init__(self, path, veh_id, X_length):
        super(featureExtract, self).__init__()
        self.path = path
        self.veh_id = veh_id
        self.X_length = X_length
        self.dataS = self.getVehicleIDData()

    def getVehicleIDData(self):
        dataS = pd.read_csv(self.path)
        frame_ori = dataS[dataS.Vehicle_ID == self.veh_id]
        GT_list = frame_ori.Global_Time.unique()
        GT_min = np.min(GT_list)
        GT_max = np.max(GT_list)

        frame_time = dataS[dataS.Global_Time >= GT_min]
        frame_time_1 = frame_time[frame_time.Global_Time <= GT_max]

        return frame_time_1

    def getData(self, veh_id):
        '''
        :param veh_id: vehicle ID
        :return: get feature data of veh_id
        '''
        AvailableTime = self.getAvailableTime(veh_id)

        print("*****Getting vehicle {} feature data*****".format(veh_id))
        length_time = len(AvailableTime)

        # All characteristic parameters have 44 dimensions in total
        # Dimension 0-5 target vehicle: (x, y, v_x, v_y, a_x, a_y)
        # Dimension 6-41 are features of surrounding vehicles
        # (left-front left-rear center-front center-rear right-front right-rear)
        # (delta_x, delta_y, v_x, v_y, a_x, a_y) * 6
        # Dimension 42-43 left and right lane flag positions

        X = 1000 * np.ones(shape=(length_time, self.X_length, 5))

        # Driving intention recognition label
        y = 1000 * np.ones(shape=(length_time, 1))

        for i in range(length_time):

            # target vehicle feature writing
            self_condition = self.getCondition(veh_id, AvailableTime[i])
            X[i, :, :] = self_condition[:, :5]            
            y[i] = int(self_condition[-1, -1])


        return X, y

    def getAvailableTime(self, veh_id):
        '''
        :param veh_id: vehicle ID
        :return: get available time of data
        '''
        dataS = self.dataS
        frame_ori = dataS[dataS.Vehicle_ID == veh_id]

        Available_time = []

        for i in range(len(frame_ori)):
            t_tmp = float(frame_ori.iloc[i, 1])
            if i >= self.X_length - 1:
                Available_time.append(t_tmp)

        return Available_time

    def getCondition(self, veh_id, t_tmp):
        '''
        :param veh_id: vehicle ID
        :param t_tmp: time stamp
        :return:
        '''
        dataS = self.dataS
        condition = np.zeros(shape=(self.X_length, 6))

        frame_ori = dataS[dataS.Vehicle_ID == veh_id]
        for i in range(self.X_length):
            frame = frame_ori[frame_ori.Global_Time == t_tmp - i]
            if not frame.empty:
                frame_history = frame
            else:
                frame = frame_history

            frame = frame[['Local_X', 'Local_Y', 'vx', 'vy', 'Lane_ID', "Lane_Change_Label"]]
            condition[i, :] = frame

        return condition



if __name__ == '__main__':

    path_in = "trajectory_Final_label.csv"

    path_X_out = "X_data.npy"
    path_y_out = "y_data.npy"

    veh_list = getTargetVehicle(path_in)

    print("*****The veh_list is:{}*****".format(len(veh_list)))

    X_length = 30

    X = []
    y = []
    for veh_id in veh_list:

        print("*****Start process veh_id {}*****".format(veh_id))
        start_time = time.time()
        FE = featureExtract(path_in, veh_id, X_length)
        X_tmp, y_tmp = FE.getData(veh_id)
        if len(y) > 0:
            X = np.vstack([X, X_tmp])
            y = np.vstack([y, y_tmp])
        else:
            X = X_tmp
            y = y_tmp

        end_time = time.time()
        print("*****End process veh_id {}*****".format(veh_id))
        print("*****time cost: {}*****".format(end_time-start_time))
        print()

        # Save the processed data into a new file
        np.save(file=path_X_out, arr=X)
        np.save(file=path_y_out, arr=y)
