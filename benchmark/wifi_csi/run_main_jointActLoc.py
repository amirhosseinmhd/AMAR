"""
[file]          run.py
[description]   run WiFi-based models
"""
#
##
import json
import argparse
from logging import raiseExceptions

import numpy as np
from gmpy2 import random_state
from sklearn.model_selection import train_test_split
#
from model import *
from preset import preset
from load_data import load_data_x, load_data_y, encode_data_y
from utils import *
#
##






def parse_args():
    """
    [description]
    : parse arguments from input
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    #
    var_args.add_argument("--model", default = preset["model"], type = str)
    var_args.add_argument("--task", default = preset["task"], type = str)
    var_args.add_argument("--repeat", default = preset["repeat"], type = int)
    var_args.add_argument("--users", default="0, 1,2,3,4,5", type=str, help="Comma-separated list of user IDs")
    #
    return var_args.parse_args()

#
##
def run():
    """
    [description]
    : run WiFi-based models
    """
    #
    ## parse arguments from input
    var_args = parse_args()
    #
    var_task = var_args.task
    var_model = var_args.model
    var_repeat = var_args.repeat
    var_users = [u.strip() for u in var_args.users.split(',')]

    preset["repeat"] = 1 if not preset["pretrained_path"] else preset["repeat"] # if we want to pretrain the model we
    #                                                                           # need only one repeat


    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment=preset["data"]["environment"],
                            var_wifi_band=preset["data"]["wifi_band"],
                            var_num_users=var_users)
    #
    var_label_list = data_pd_y["label"].to_list()
    #
    ## load CSI amplitude
    X = load_data_x(preset["path"]["data_x"], var_label_list)

    y_activity = encode_data_y(data_pd_y, "activity")
    y_location = encode_data_y(data_pd_y, "location")
    if preset["model"] == "multiSense_X":
        y_activity_n, y_location_n = reduce_dataset_joint_multiSenseX(y_activity, y_location)
    else:
        y_activity_n, y_location_n = reduce_dataset_joint(y_activity, y_location, preset["nn"]["num_obj_queries"])

    (X_train, X_test,
     y_train_loc, y_test_loc,
     y_train_act, y_test_act) = my_train_test_split(X, y_location_n, y_activity_n, test_size=0.2, random_state=103)

    if preset["model"] == "multiSense_X":
        run_model  = run_multi_senseX
    if preset["model"] == "joint_AMAR":
        run_model = run_joint_AMAR

    result_act, result_loc = run_model(X_train, y_train_loc, y_train_act,
                                                X_test, y_test_loc, y_test_act,
                                                var_repeat)
    #
    ##
    # result["model"] = var_model
    # result["task"] = var_task
    # result["data"] = preset["data"]
    # result["nn"] = preset["nn"]
    # #
    print(result_act)
    print(result_loc)
    # #
    # ## save results
    # var_file = open(preset["path"]["save"], 'w')
    # json.dump(result, var_file, indent=4, cls=NumpyEncoder)

#
##

if __name__ == "__main__":
    #
    ##
    run()