"""
[file]          run.py
[description]   run WiFi-based models
"""
#
##
import json
import argparse
from sklearn.model_selection import train_test_split
#
from model import *
from preset import preset
from load_data import load_data_x, load_data_y, encode_data_y
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
    var_args.add_argument("--users", default="0,1,2,3,4,5", type=str, help="Comma-separated list of user IDs")
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
    # var_users = [u.strip() for u in var_args.users.split(',')]
    # preset["data"]["num_users"]  = var_users
    #
    ## load annotation file as labels
    data_pd_y_band1 = load_data_y(preset["path"]["data_y"],
                                  var_environment = preset["data"]["environment"],
                                  var_wifi_band = preset["data"]["wifi_band"],
                                  var_num_users = preset["data"]["num_users"])

    data_pd_y_band2 = load_data_y(preset["path"]["data_y"],
                            var_environment=preset["data_band2"]["environment"],
                            var_wifi_band=preset["data_band2"]["wifi_band"],
                            var_num_users=preset["data_band2"]["num_users"])


    #
    var_label_list_band1 = data_pd_y_band1["label"].to_list()
    var_label_list_band2 = data_pd_y_band2["label"].to_list()

    #
    ## load CSI amplitude
    data_x_band1 = load_data_x(preset["path"]["data_x"], var_label_list_band1)
    data_x_band2 = load_data_x(preset["path"]["data_x"], var_label_list_band2)
    #
    ## encode labels
    data_y = encode_data_y(data_pd_y_band1, var_task)
    data_y_band2 = encode_data_y(data_pd_y_band2, var_task)
    #
    ## a training set (80%) and a test set (20%)
    data_train_x_band1, data_test_x_band1, data_train_y_band1, data_test_y_band1 = train_test_split(
                                                                                            data_x_band1,
                                                                                                    data_y,
                                                                                                    test_size = 0.2,
                                                                                                    shuffle = True,
                                                                                                    random_state = 39)
    data_train_x_band2, data_test_x_band2, data_train_y_band2, data_test_y_band2 = train_test_split(
                                                        data_x_band2,
                                                                data_y_band2,
                                                             test_size=0.2,
                                                             shuffle=True,
                                                             random_state=39
                                                             )




    #
    ## run WiFi-based model
    result = run_dual_band(data_train_x_band1, data_train_y_band1,
                       data_test_x_band1, data_test_y_band1,
                       data_train_x_band2, data_train_y_band2,
                       data_test_x_band2, data_test_y_band2,
                       var_repeat)
    #
    ##
    result["model"] = var_model
    result["task"] = var_task
    result["data"] = preset["data"]
    result["nn"] = preset["nn"]
    #
    print(result)
    #
    ## save results
    var_file = open(preset["path"]["save"], 'w')
    json.dump(result, var_file, indent = 4)

#
##
if __name__ == "__main__":
    #
    ##
    run()