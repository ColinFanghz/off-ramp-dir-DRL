"""
@Author: Fhz
@Create Date: 2023/10/17 16:39
@File: sac_dir_test.py
@Description:
@Modify Person Date:
"""
from env_dir_test import *
from stable_baselines3 import SAC
import argparse


def get_returns(arr, gamma):

    returns = 0

    for i in range(len(arr)):
        returns = returns + arr[i] * gamma ** i

    return returns

def get_args():
    parser = argparse.ArgumentParser()
    # SUMO config
    parser.add_argument("--count", type=int, default=0, help="The length of a training episode.")
    parser.add_argument("--show_gui", type=bool, default=False, help="The flag of show SUMO gui.")
    parser.add_argument("--sumocfgfile", type=str, default="../sumo_config/my_config_file.sumocfg", help="The path of the SUMO configure file.")
    parser.add_argument("--egoID", type=str, default="self_car", help="The ID of ego vehicle.")
    parser.add_argument("--start_time", type=int, default=240, help="The simulation step before learning.")
    parser.add_argument("--collision", type=bool, default=False, help="The flag of collision of ego vehicle.")
    parser.add_argument("--sleep", type=bool, default=True, help="The flag of sleep of each simulation.")
    parser.add_argument("--y_none", type=float, default=2000.0, help="The longitudinal position of a none exist vehicle.")
    parser.add_argument("--vehicle_default_length", type=float, default=5.0, help="The default length of vehicle.")
    parser.add_argument("--vehicle_default_width", type=float, default=2.4, help="The default width of vehicle.")
    parser.add_argument("--num_action", type=int, default=142, help="The number of action space.")
    parser.add_argument("--lane_change_time", type=float, default=1.0, help="The time of lane change.")
    parser.add_argument("--prob_main", type=float, default=0.45, help="The probability of main lane.")
    parser.add_argument("--prob_merge", type=float, default=0.05, help="The probability of merge lane.")
    parser.add_argument("--seed_value", type=str, default="123", help="The seed value.")
    parser.add_argument("--seed_value1", type=str, default="456", help="The seed value1.")
    parser.add_argument("--gamma", type=str, default="0.999", help="The reward param.")
    parser.add_argument("--t_sample", type=float, default="0.1", help="The reward param.")
    parser.add_argument("--model_name", type=str, default="../DIR/xgboost.dat", help="The driving intention recognition model.")

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
    parser.add_argument("--gap", type=float, default=10.0, help="The threshold of ego vehicle to other vehicle.")
    parser.add_argument("--leaderMaxDecel", type=float, default=4.5, help="The leader maximum deceleration.")

    # Reward config
    parser.add_argument("--w_jerk_x", type=float, default=0.005, help="The weight of lateral jerk reward.")
    parser.add_argument("--w_jerk_y", type=float, default=0.005, help="The weight of longitudinal jerk reward.")
    parser.add_argument("--w_time", type=float, default=0.1, help="The weight of time consuming reward.")
    parser.add_argument("--w_lane", type=float, default=2, help="The weight of target lane reward.")
    parser.add_argument("--w_speed", type=float, default=0.1, help="The weight of desired speed reward.")
    parser.add_argument("--R_time", type=float, default=-0.1, help="The reward of time consuming.")
    parser.add_argument("--V_desired", type=float, default=20.0, help="The desired speed.")
    parser.add_argument("--R_collision", type=float, default=-400, help="The reward of ego vehicle collision.")
    parser.add_argument("--P_left", type=float, default=9.0, help="The lateral position of target lane.")
    parser.add_argument("--P_target", type=float, default=1.8, help="The lateral position of target lane.")
    parser.add_argument("--P_za", type=float, default=1.8, help="The lateral position of za lane.")
    parser.add_argument("--lane_width", type=float, default=3.6, help="The width of a lane.")
    parser.add_argument("--mainRouteProb", type=float, default=0.9, help="The width of a lane.")

    # Done config
    parser.add_argument("--target_lane_id", type=int, default=1, help="The ID of target lane.")
    parser.add_argument("--merge_position", type=float, default=410.00, help="The position of the merge lane.")
    parser.add_argument("--max_count", type=int, default=60, help="The maximum length of a training episode.")

    # save train data
    parser.add_argument("--train_flag", type=bool, default=False, help="The flag of training.")
    parser.add_argument("--total_reward", type=str, default="totalReward.npy", help="The total reward of training.")
    parser.add_argument("--comfort_reward", type=str, default="comfortReward.npy", help="The comfort reward of training.")
    parser.add_argument("--efficiency_reward", type=str, default="efficiencyReward.npy", help="The efficiency reward of training.")
    parser.add_argument("--safety_reward", type=str, default="safetyReward.npy", help="The safety reward of training.")
    parser.add_argument("--total_loss", type=str, default="totalLoss.npy", help="The total loss of training.")
    parser.add_argument("--success_rate", type=str, default="successRate.npy", help="The success rate of training.")
    parser.add_argument("--collision_rate", type=str, default="collisionRate.npy", help="The collision rate of training.")


    parser.add_argument("--input_size", type=int, default=6, help="The size of input.")
    parser.add_argument("--hidden_size", type=int, default=128, help="The hidden size.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of training process.")
    parser.add_argument("--target_len", type=int, default=10, help="The target trajectory length.")
    parser.add_argument("--teacher_rate", type=float, default=0.5, help="The teacher rate.")

    parser.add_argument("--trajectory_length", type=int, default=30, help="The length of trajectory.")
    parser.add_argument("--feature_length", type=int, default=6, help="The length of feature.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    env = SumoGym(args)

    global successCount
    global collisionCount
    global episode
    global total_rewards
    global comfort_rewards
    global efficiency_rewards
    global safety_rewards
    global num_itr
    global total_counts
    global speeds
    global gamma
    global results

    successCount = 0
    collisionCount = 0
    episode = 0
    total_rewards = []
    comfort_rewards = []
    efficiency_rewards = []
    safety_rewards = []
    num_itr = 0
    total_counts = 0
    speeds = []
    gamma = 0.999
    results = []


    models = ["sac1_dir", "sac2_dir", "sac3_dir"]

    model_results = []
    for model_tmp in models:
        model = SAC.load(model_tmp, env=env)

        gamma = 0.999
        eposides = 200
        rewards = []
        speeds = []
        success_count = 0
        collision_count = 0
        counts = 0

        for eq in range(eposides):
            print("Test eposide: {}".format(eq))
            obs = env.reset()
            env.seed_value = "123{}".format(eq)
            env.seed_value1 = "456{}".format(eq)

            done = False

            count = 0
            reward_tmp = []
            speed_tmp = 0
            while not done:
                count += 1
                time.sleep(0.01)
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

    np.save("model_results/sac_dir_123_results.npy", arr=np.array(model_results))