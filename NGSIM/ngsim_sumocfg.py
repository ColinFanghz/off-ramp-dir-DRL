"""
@Author: Fhz
@Create Date: 2023/7/25 20:59
@File: ngsim_sumocfg.py
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


def get_config_file(self_vehicle_ID):
    """
    :param self_vehicle_ID: ego vehicle ID
    :param scene: scene data
    :return:
    """
    # 创建根元素
    root = ET.Element("configuration")

    # 添加子元素和属性
    input1 = ET.SubElement(root, "input")

    netFile = ET.SubElement(input1, "net-file")
    netFile.set('value', "../ngsim1.net.xml")
    netFile = ET.SubElement(input1, "route-files")
    netFile.set('value', "../routes/data1317/route{}.rou.xml".format(self_vehicle_ID))

    time1 = ET.SubElement(root, "time")
    begin = ET.SubElement(time1, "begin")
    begin.set('value', "0")
    end = ET.SubElement(time1, "end")
    end.set('value', "1000")
    stepLength = ET.SubElement(time1, "step-length")
    stepLength.set('value', "0.1")

    guiOnly = ET.SubElement(root, "gui_only")
    start = ET.SubElement(guiOnly, "start")
    start.set('value', "t")
    quitOnEnd = ET.SubElement(guiOnly, "quit-on-end")
    quitOnEnd.set('value', "t")

    processing = ET.SubElement(root, "processing")
    LC = ET.SubElement(processing, "lanechange.duration")
    LC.set('value', "6")

    # 创建 XML 树并保存到文件
    tree = ET.ElementTree(root)
    __indent(root)
    tree.write("ngsim_config/sumocfg/data1317/config_file{}.sumocfg".format(self_vehicle_ID), )


def get_args():
    parser = argparse.ArgumentParser()
    # SUMO config
    # parser.add_argument("--dataset", type=str, default="datasets/trajectories-0750am-0805am.csv",
    #                     help="The path of the dataset.")
    # parser.add_argument("--dataset", type=str, default="datasets/trajectories-0805am-0820am.csv",
    #                     help="The path of the dataset.")
    parser.add_argument("--dataset", type=str, default="datasets/trajectories-0820am-0835am.csv",
                        help="The path of the dataset.")

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
        get_config_file(veh_id)
