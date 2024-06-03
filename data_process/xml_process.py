"""
@Author: Fhz
@Create Date: 2023/7/13 13:55
@File: xml_process.py
@Description: 
@Modify Person Date: 
"""
from xml.dom import minidom
import csv
import math


# 打开xml文档
dom = minidom.parse('../dataset_generation/fcd.xml')  # parse用于打开一个XML文件
# 得到文档元素对象
root = dom.documentElement  # documentElement用于得到XML文件的唯一根元素


f = open('trajectory.csv', 'w', newline="", encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "Heading_Angle", "vx", "vy", "Vtype", "Lane_ID", "Lane_Change_Label"])
# x-lateral, y-longitudinal

timesteps = root.getElementsByTagName('timestep')

ids = 1
sim_during = 12000

for ii in range(len(timesteps)):
    timestep = timesteps[ii]
    vehicles = timestep.getElementsByTagName('vehicle')

    for vehicle in vehicles:
        ID = vehicle.getAttribute("id")
        j = int(ID[5])
        k = float(ID[7:])

        print(ID)

        y = float(vehicle.getAttribute("x"))
        x = float(vehicle.getAttribute("y"))
        heading = float(vehicle.getAttribute("angle"))
        speed = float(vehicle.getAttribute("speed"))
        angle = (math.pi * (heading - 90) / 180)
        vx = speed * math.sin(angle)
        vy = speed * math.cos(angle)

        if vehicle.getAttribute("type") == "car":
            Vtype = 0
        elif vehicle.getAttribute("type") == "motorcycle":
            Vtype = 1
        elif vehicle.getAttribute("type") == "truck":
            Vtype = 2
        else:
            print("Error vehicle type!")

        if y >= 18.0:
            lane_id = 1
        elif y >= 14.4:
            lane_id = 2
        elif y >= 10.8:
            lane_id = 3
        elif y >= 7.2:
            lane_id = 4
        elif y >= 3.6:
            lane_id = 5
        elif y < 3.6:
            if x < 157.02:
                lane_id = 7
            elif x > 412.77:
                lane_id = 8
            else:
                lane_id = 6
        else:
            print("Error lane ID!")


        csv_writer.writerow([sim_during*j+k, ii, x, y, angle, vx, vy, Vtype, lane_id, 1])
        ids += 1

f.close()