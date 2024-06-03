"""
@Author: Fhz
@Create Date: 2022/11/6 20:48
@File: preprocess.py
@Description: 
@Modify Person Date: 
"""
import pandas as pd
import numpy as np
import math
import csv

"""
class preprocess():
    def __init__(self, path_ori, path_final):
        super(preprocess, self).__init__()
        self.path_ori = path_ori
        self.path_final = path_final
        self.laneChangeLable = self.getLaneChangeLabel()
        


    def getLaneChangeLabel(self):
        '''
        :return: Add lane change label
        '''

        dataS = pd.read_csv(self.path_ori)
        max_vehiclenum = np.max(dataS.Vehicle_ID.unique())
        max_vehiclenum = int(max_vehiclenum)
        print(max_vehiclenum)
        thd = 0.1

        # Store label data
        label_storage = []

        for i in range(max_vehiclenum + 1):
            frame_ori = dataS[dataS.Vehicle_ID == i]
            if len(frame_ori) == 0:
                continue

            t_first = np.min(frame_ori.Global_Time.unique())
            print("Vehicle ID: {}, length of data: {}".format(i, len(frame_ori)))

            lane_change_time = []  # lane change time stamp
            t_history = t_first  # history lane change time stamp
            for j in range(len(frame_ori) - 1):
                t_tmp = t_first + j
                frame = frame_ori[frame_ori.Global_Time == t_tmp]
                frame_1 = frame_ori[frame_ori.Global_Time == t_tmp + 1]

                lane_id = float(frame.Lane_ID)
                lane_id_1 = float(frame_1.Lane_ID)
                label_end = 1

                # Store lane change time stamp
                if lane_id > lane_id_1:  # left lane change
                    print("Vehicle ID: {}, time stamp: {}, lane change label: {}".format(i, t_tmp, 0))
                    lane_change_time.append([t_history, t_tmp, 0])
                    t_history = t_tmp
                    label_end = 0
                elif lane_id < lane_id_1:
                    print("Vehicle ID: {}, time stamp: {}, lane change label: {}".format(i, t_tmp, 2))
                    lane_change_time.append([t_history, t_tmp, 2])
                    t_history = t_tmp
                    label_end = 2

            lane_change_time.append([t_history, t_first + len(frame_ori) - 1, label_end])

            if len(lane_change_time) == 1:
                continue
            else:
                ### lane_change_time: First point, index from back to front
                t0, t1, label0 = lane_change_time[0]
                t0 = int(t0)
                t1 = int(t1)

                # Reduce the area within 40 steps
                if t1 - t0 > 40:
                    t0 = t1 - 40

                count_heading = 0
                if label0 == 0:
                    for tmp in range(t1, t0 - 1, -1):
                        frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                        if frame_heading.iloc[0, 7] > -thd:  # left heading angle threshold
                            count_heading = count_heading + 1
                            if count_heading >= 3:
                                break
                            else:
                                label_storage.append([i, tmp, label0])
                        else:
                            label_storage.append([i, tmp, label0])
                            count_heading = 0

                elif label0 == 2:
                    for tmp in range(t1 + 1, t0 - 1, -1):
                        frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                        if frame_heading.iloc[0, 7] < thd:  # right heading angle threshold
                            count_heading = count_heading + 1
                            if count_heading >= 3:
                                break
                            else:
                                label_storage.append([i, tmp, label0])
                        else:
                            label_storage.append([i, tmp, label0])
                            count_heading = 0

                ### lane_change_time: middle point
                if len(lane_change_time) > 2:
                    for o in range(1, len(lane_change_time) - 1):
                        t_0_no_use, t_1_no_use, label_0 = lane_change_time[o - 1]
                        t_0, t_1, label_1_no_use = lane_change_time[o]

                        t_0 = int(t_0)
                        t_1 = int(t_1)
                        # Reduce the area within 40 steps
                        # Front half area, indexed from front to back
                        if t_1 - t_0 > 40:
                            t1 = t_0 + 40

                        count_heading = 0
                        if label_0 == 0:
                            for tmp in range(t_0, t1 + 1):
                                frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                                if frame_heading.iloc[0, 7] > -thd:
                                    count_heading = count_heading + 1
                                    if count_heading >= 3:
                                        break
                                    else:
                                        label_storage.append([i, tmp, label0])
                                else:
                                    label_storage.append([i, tmp, label0])
                                    count_heading = 0

                        elif label_0 == 2:
                            for tmp in range(t_0, t1 + 1):
                                frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                                if frame_heading.iloc[0, 7] < thd:
                                    count_heading = count_heading + 1
                                    if count_heading >= 3:
                                        break
                                    else:
                                        label_storage.append([i, tmp, label0])
                                else:
                                    label_storage.append([i, tmp, label0])
                                    count_heading = 0

                        # Reduce the area within 40 steps
                        # Second half area, indexed from back to front
                        if t_1 - t_0 > 40:
                            t0 = t_1 - 40

                        count_heading = 0
                        if label_0 == 0:
                            for tmp in range(t_1, t0 - 1, -1):
                                frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                                if frame_heading.iloc[0, 7] > -thd:
                                    count_heading = count_heading + 1
                                    if count_heading >= 3:
                                        break
                                    else:
                                        label_storage.append([i, tmp, label0])
                                else:
                                    label_storage.append([i, tmp, label0])
                                    count_heading = 0

                        elif label_0 == 2:
                            for tmp in range(t_1, t0 - 1, -1):
                                frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                                if frame_heading.iloc[0, 7] < thd:
                                    count_heading = count_heading + 1
                                    if count_heading >= 3:
                                        break
                                    else:
                                        label_storage.append([i, tmp, label0])
                                else:
                                    label_storage.append([i, tmp, label0])
                                    count_heading = 0

                ### lane_change_time: Final point
                t_0_no_use, t_1_no_use, label_0 = lane_change_time[len(lane_change_time) - 2]
                t_0, t_1, label_1_no_use = lane_change_time[len(lane_change_time) - 1]

                t_0 = int(t_0)
                t_1 = int(t_1)

                # Reduce the area within 40 steps, Front half area
                if t_1 - t_0 > 40:
                    t1 = t_0 + 40

                count_heading = 0
                if label_0 == 0:
                    for tmp in range(t_0, t1 + 1):
                        frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                        if frame_heading.iloc[0, 7] > -thd:
                            count_heading = count_heading + 1
                            if count_heading >= 3:
                                break
                            else:
                                label_storage.append([i, tmp, label0])
                        else:
                            label_storage.append([i, tmp, label0])
                            count_heading = 0

                elif label_0 == 2:
                    for tmp in range(t_0, t1 + 1):
                        frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                        if frame_heading.iloc[0, 7] < thd:
                            count_heading = count_heading + 1
                            if count_heading >= 3:
                                break
                            else:
                                label_storage.append([i, tmp, label0])
                        else:
                            label_storage.append([i, tmp, label0])
                            count_heading = 0

            lane_change_time = []

        # Remove duplicate data
        label_storage_new = []
        for label_tmp in label_storage:
            if label_tmp not in label_storage_new:
                label_storage_new.append(label_tmp)

        data_new = pd.DataFrame(columns=["Vehicle_ID", "Global_Time", "lane_change_label"])
        data_tmp = pd.DataFrame(columns=["Vehicle_ID", "Global_Time", "lane_change_label"])

        for ii in range(len(label_storage_new)):
            data_tmp.loc[1] = label_storage[ii]
            data_new = data_new.append(data_tmp, ignore_index=True)

        return data_new

    def replaceLabel(self):
        '''
        :return: replace "xxx_addLabel.csv" lane change label with "xxx_label.csv".
                 store the result to new file "xxx_Final_label.csv".
        '''

        dataS = pd.read_csv(self.path_ori)
        dataS_1 = self.laneChangeLable
        print(dataS_1.values)

        ID_lists = dataS_1.Vehicle_ID.unique()

        f = open(self.path_final, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Vehicle_ID",
                             "Global_Time",
                             "Local_X",
                             "Local_Y",
                             "vx",
                             "vy",
                             "Heading_Angle",
                             "Vtype",
                             "Lane_ID",
                             "Lane_Change_Label"])

        for i in range(len(dataS)):
            dataS_tmp = dataS.iloc[i, :]
            veh_id = dataS_tmp.iloc[0]
            if veh_id not in ID_lists:
                csv_writer.writerow(dataS_tmp)
            else:
                time_tmp = dataS_tmp.iloc[1]
                dataS_1_tmp = dataS_1[dataS_1.Vehicle_ID == veh_id]
                dataS_1_tmp1 = dataS_1_tmp[dataS_1_tmp.Global_Time == time_tmp]
                if len(dataS_1_tmp1) == 0:
                    csv_writer.writerow(dataS_tmp)
                else:
                    csv_writer.writerow([dataS_tmp.iloc[0],
                                         dataS_tmp.iloc[1],
                                         dataS_tmp.iloc[2],
                                         dataS_tmp.iloc[3],
                                         dataS_tmp.iloc[4],
                                         dataS_tmp.iloc[5],
                                         dataS_tmp.iloc[6],
                                         dataS_tmp.iloc[7],
                                         dataS_tmp.iloc[8],
                                         dataS_1_tmp1.iloc[0, 2]])

            if i % 10000 == 0:
                print("Written: {}".format(i))

        f.close()
"""    
    
def newLabel(path_ori, path_final):
    dataS = pd.read_csv(path_ori)
    
    f = open(path_final, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Vehicle_ID",
                             "Global_Time",
                             "Local_X",
                             "Local_Y",
                             "Heading_Angle",
                             "vx",
                             "vy",
                             "Vtype",
                             "Lane_ID",
                             "Lane_Change_Label"])
    
    for i in range(len(dataS)):
        dataS_tmp = dataS.iloc[i, :]
        Heading_Angle = dataS_tmp.iloc[4]
        Local_X = dataS_tmp.iloc[2]
        if Heading_Angle > 0:
            csv_writer.writerow([dataS_tmp.iloc[0],
                                         dataS_tmp.iloc[1],
                                         dataS_tmp.iloc[2],
                                         dataS_tmp.iloc[3],
                                         dataS_tmp.iloc[4],
                                         dataS_tmp.iloc[5],
                                         dataS_tmp.iloc[6],
                                         dataS_tmp.iloc[7],
                                         dataS_tmp.iloc[8],
                                         2])
        elif Heading_Angle < 0 and Local_X >= 3.6:
            csv_writer.writerow([dataS_tmp.iloc[0],
                                         dataS_tmp.iloc[1],
                                         dataS_tmp.iloc[2],
                                         dataS_tmp.iloc[3],
                                         dataS_tmp.iloc[4],
                                         dataS_tmp.iloc[5],
                                         dataS_tmp.iloc[6],
                                         dataS_tmp.iloc[7],
                                         dataS_tmp.iloc[8],
                                         0])
        else:
            csv_writer.writerow(dataS_tmp)
            
        if i % 10000 == 0:
                print("Written: {}".format(i))
        
    f.close()
    


if __name__ == '__main__':

    path_ori = "trajectory.csv"

    path_final = "trajectory_Final_label.csv"

    # Pre = preprocess(path_ori, path_final)
    # Pre.replaceLabel()
    newLabel(path_ori, path_final)