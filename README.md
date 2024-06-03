# off-ramp-dir-DRL
code for paper "Robust Decision-Making for Off-Ramp in Intention-based Deep Reinforcement Learning". 

# User Guide

Generate training dataset：
run dataset_generation/main.py --> generation sumo dataset.

Data process:
run data_process/xml_process.py --> transfer the data from xml into csv.
run data_process/addLaneChangeLabel.py --> add lane change label to the dataset.
run data_process/final_DP.py --> generate the final dataset.
run data_process/dataset_divide.py --> divide the dataset into train and test set.

Driving intention recognition training process:
run DIR/XGBoost.py --> driving intention recognition through xgboost.
run DIR/RF.py --> driving intention recognition through random forest.

Deep reinforcement learning training:
run DRL/sac_train.py and DRL/sac_dir_train.py --> Train deep reinforcement learning model.
run DRL/sac_test.py and DRL/sac_dir_test.py --> Test deep reinforcement learning model.

Valid in NGSIM datasets
run NGSIM/python ngsim_routes.py   --> Generate route file.
run NGSIM/python ngsim_sumocfg.py,--> Generate config file.
run NGSIM/python ngsim_replay.py, --> Generate the whole replay file of vehicles.
run NGSIM/python ngsim_TV_replay.py, --> Generate the whole replay file of target vehicles.
run NGSIM/python gym_ngsim_sac_off_dir_test.py --> Test deep reinforcement learning model in NGSIM.

