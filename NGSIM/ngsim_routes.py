"""
@Author: Fhz
@Create Date: 2023/7/25 16:06
@File: ngsim_routes.py
@Description: 
@Modify Person Date: 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import argparse


def get_all_vehicle_IDs(dataS):
    """
    :param dataS: The pandas datasets
    :return: The ID of all the off ramp vehicles
    """
    # frame = dataS[dataS.Lane_ID == 8]
    Vehicle_IDs = dataS.Vehicle_ID.unique()

    return Vehicle_IDs


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


def plot_trajectory(vehicle_ID, frame):
    """
    :param vehicle_ID:  vehicle ID
    :param frame:  The frame with vehicle ID
    :return: None
    """
    frame_ID = frame[frame.Vehicle_ID == vehicle_ID]
    frame_ID_new = unitConversion(frame_ID)
    frame_ID_np = frame_ID_new.values
    plt.plot(frame_ID_np[:, 5], frame_ID_np[:, 4], label="Vehicle_ID:{}".format(vehicle_ID))
    plt.legend()
    plt.show()


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


def get_scene(vehicle_ID, dataS):
    """
    :param vehicle_ID: Vehicle ID
    :param dataS: dataset
    :return:
    """
    frame = dataS[dataS.Vehicle_ID == vehicle_ID]

    t_start = np.min(frame.Global_Time.unique())
    t_end = np.max(frame.Global_Time.unique())

    # 限制时间
    frame_1 = dataS[dataS.Global_Time >= t_start]
    frame_vehicle = frame_1[frame_1.Global_Time <= t_end]

    # 限制初始位置
    Y_boundary = 50 / 0.3048
    vehicle_IDs = []
    for ii in range(len(frame)):
        t_tmp = t_start + ii * 100
        frame_vehicle_tmp = frame_vehicle[frame_vehicle.Global_Time == t_tmp]
        frame_vehicle_ego_tmp = frame_vehicle_tmp[frame_vehicle_tmp.Vehicle_ID == vehicle_ID]
        ego_Local_y = np.min(frame_vehicle_ego_tmp.Local_Y.unique())
        frame_vehicle1 = frame_vehicle_tmp[frame_vehicle_tmp.Local_Y <= ego_Local_y + Y_boundary]
        frame_vehicle_fin = frame_vehicle1[frame_vehicle1.Local_Y >= ego_Local_y - Y_boundary]
        vehicle_ID_tmp = frame_vehicle_fin.Vehicle_ID.unique().tolist()
        vehicle_IDs = vehicle_IDs + vehicle_ID_tmp

    frame_vehicle_IDs = list(set(vehicle_IDs))



    # # 不限制位置
    # frame_vehicle_IDs = frame_vehicle.Vehicle_ID.unique()

    scene = np.zeros([len(frame_vehicle_IDs), 7])

    for i in range(len(frame_vehicle_IDs)):
        veh_id = frame_vehicle_IDs[i]
        frame_veh_id = frame_vehicle[frame_vehicle.Vehicle_ID == veh_id]
        frame_veh_id = unitConversion(frame_veh_id)

        t_start_tmp = np.min(frame_veh_id.Global_Time.unique())
        t_end_tmp = np.max(frame_veh_id.Global_Time.unique())
        frame_start = frame_veh_id[frame_veh_id.Global_Time == t_start_tmp].values
        frame_end = frame_veh_id[frame_veh_id.Global_Time == t_end_tmp].values
        lane_start = frame_start[0, 13]
        veh_type = frame_start[0, 10]
        Local_y_start = frame_start[0, 5]
        V_speed = frame_start[0, 11]
        lane_end = frame_end[0, 13]

        route_ID = get_route_ID(lane_start, lane_end, Local_y_start)

        if veh_id == vehicle_ID:
            depart_time = 0
        else:
            depart_time = 0.001 * (t_start_tmp - t_start)

        scene[i, 0] = veh_id
        scene[i, 1] = lane_start
        scene[i, 2] = round(Local_y_start, 2)
        scene[i, 3] = round(V_speed, 2)
        scene[i, 4] = route_ID
        scene[i, 5] = round(depart_time,1)
        scene[i, 6] = veh_type

    scene = scene[scene[:, 5].argsort()]

    return scene


def __indent(elem, level=0):
    """
    :param elem:
    :param level:
    :return: Save config
    """
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def get_config_file(self_vehicle_ID,scene):
    """
    :param self_vehicle_ID: ego vehicle ID
    :param scene: scene data
    :return:
    """
    root = ET.Element('routes')  # 创建节点
    tree = ET.ElementTree(root)  # 创建文档

    vehicle_type = ["none", "motorcycle", "car", "truck"]

    data = [
        {"id":"route1.0", "edges":"gneE0 gneE1 gneE2"},
        {"id":"route2.0", "edges":"gneE0 gneE1 gneE5"},
        {"id":"route3.0", "edges":"gneE3 gneE1 gneE2"},
        {"id":"route4.0", "edges":"gneE3 gneE1 gneE5"},
        {"id":"route5.0", "edges":"gneE1 gneE2"},
        {"id":"route6.0", "edges":"gneE1 gneE5"},
    ]

    data1 = [
        {"id": "motorcycle",
         "accel": "4.0",
         "decel": "4.5",
         "emergencyDecel": "9.0",
         "length": "2",
         "color": "0,1,1",
         "vClass": "motorcycle",
         "maxSpeed": "40",
         "carFollowModel": "IDM",
         "actionStepLength": "0.1"},

        {"id": "car",
         "accel": "2.5",
         "decel": "4.5",
         "emergencyDecel": "9.0",
         "length": "5",
         "color": "0,1,0",
         "vClass": "passenger",
         "maxSpeed": "35",
         "carFollowModel": "IDM",
         "actionStepLength": "0.1"},

        {"id": "truck",
         "accel": "1.8",
         "decel": "4.5",
         "emergencyDecel": "9.0",
         "length": "15",
         "color": "1,0,0",
         "vClass": "truck",
         "maxSpeed": "30",
         "carFollowModel": "IDM",
         "actionStepLength": "0.1"},
    ]

    for item in data1:
        row_elem = ET.SubElement(root, 'vType')
        row_elem.set('id', item['id'])
        row_elem.set('accel', item['accel'])
        row_elem.set('decel', item['decel'])
        row_elem.set('emergencyDecel', item['emergencyDecel'])
        row_elem.set('length', item['length'])
        row_elem.set('color', item['color'])
        row_elem.set('vClass', item['vClass'])
        row_elem.set('maxSpeed', item['maxSpeed'])
        row_elem.set('carFollowModel', item['carFollowModel'])
        row_elem.set('actionStepLength', item['actionStepLength'])

    for item in data:
        row_elem = ET.SubElement(root, 'route')
        row_elem.set('id', item['id'])
        row_elem.set('edges', item['edges'])

    datas = []

    for i in range(len(scene)):
        veh_id = scene[i, 0]
        departSpeed = scene[i, 3]

        # route1,2
        if scene[i, 4] <= 2:
            departPos = scene[i, 2]
            route = "route{}".format(scene[i, 4])
            if scene[i, 1] == 1:
                departlane = "4"
            elif scene[i, 1] == 2:
                departlane = "3"
            elif scene[i, 1] == 3:
                departlane = "2"
            elif scene[i, 1] == 4:
                departlane = "1"
            else:
                departlane = "0"
        # route3,4
        elif scene[i, 4] <= 4:
            departPos = scene[i, 2]
            route = "route{}".format(scene[i, 4])
            departlane = "0"
        # route5,6
        else:
            departPos = scene[i, 2] - 176.10
            route = "route{}".format(scene[i, 4])
            if scene[i, 1] == 1:
                departlane = "5"
            elif scene[i, 1] == 2:
                departlane = "4"
            elif scene[i, 1] == 3:
                departlane = "3"
            elif scene[i, 1] == 4:
                departlane = "2"
            elif scene[i, 1] == 5:
                departlane = "1"
            else:
                departlane = "0"
        print(
            "<vehicle id=\"{}\" depart=\"{}\" departLane=\"{}\" departPos=\"{}\" departSpeed=\"{}\" route=\"{}\" type=\"{}\"/>".format(
                veh_id, scene[i, 5], departlane, departPos, departSpeed, route, vehicle_type[int(scene[i, 6])]))
        datas.append({"id": "{}".format(veh_id),
                         "depart": "{}".format(scene[i, 5]),
                         "departLane": "{}".format(departlane),
                         "departPos": "{}".format(departPos),
                         "departSpeed": "{}".format(departSpeed),
                         "route": "{}".format(route),
                         "type": "{}".format(vehicle_type[int(scene[i, 6])]),
                         # "color": "0,1,1",
                         })

    for item in datas:
            row_elem = ET.SubElement(root, 'vehicle')
            row_elem.set('id', item['id'])
            row_elem.set('depart', item['depart'])
            row_elem.set('departLane', item['departLane'])
            row_elem.set('departPos', item['departPos'])
            row_elem.set('departSpeed', item['departSpeed'])
            row_elem.set('route', item['route'])
            row_elem.set('type', item['type'])
            # row_elem.set('color', item['color'])

    tree = ET.ElementTree(root)
    __indent(root)
    tree.write('ngsim_config/sumocfg/routes/data2783/route{}.rou.xml'.format(self_vehicle_ID))


def get_args():
    parser = argparse.ArgumentParser()
    # SUMO config
    parser.add_argument("--show_gui", type=bool, default=True, help="The flag of show SUMO gui.")
    parser.add_argument("--sumocfgfile", type=str, default="ngsim_config/my_config_file.sumocfg",
                        help="The path of the SUMO configure file.")
    parser.add_argument("--dataset", type=str, default="datasets/trajectories-0750am-0805am.csv",
                        help="The path of the dataset.")
    # parser.add_argument("--dataset", type=str, default="datasets/trajectories-0805am-0820am.csv",
    #                     help="The path of the dataset.")
    # parser.add_argument("--dataset", type=str, default="datasets/trajectories-0820am-0835am.csv",
    #                     help="The path of the dataset.")
    parser.add_argument("--routes_files", type=str, default="ngsim_config/routes",
                        help="The route config path.")

    args = parser.parse_args()

    return args


def getOnMergeVehicleID(frame):
    frameM = frame[frame.Lane_ID == 7]

    vehicle_ID_List = frameM.Vehicle_ID.unique().tolist()

    return vehicle_ID_List


def getOffMergeVehicleID(frame):

    frameM = frame[frame.Lane_ID == 8]

    vehicle_ID_List = frameM.Vehicle_ID.unique().tolist()

    return vehicle_ID_List


if __name__ == '__main__':
    args = get_args()

    path = args.dataset
    dataS = pd.read_csv(path)
    # IDS = getOnMergeVehicleID(dataS)
    IDS = getOffMergeVehicleID(dataS)
    for veh_id in IDS:
        scene = get_scene(veh_id, dataS)
        get_config_file(veh_id, scene)
        # break