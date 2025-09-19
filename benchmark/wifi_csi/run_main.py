"""
[file]          run.py
[description]   run WiFi-based models
"""
#
##

import argparse
import random # Added
from sklearn.model_selection import train_test_split
from model import *
from load_data import load_data_x, load_data_y, encode_data_y
from utils import *
from preset import preset

#
##
def master_splitter(preset, var_task, var_model, var_users):
    env_data_x_train = []
    env_data_x_test = []
    env_data_y_train = []
    env_data_y_test = []
    for env in preset["data"]["environment"]:
        data_pd_y = load_data_y(preset["path"]["data_y"],
                                var_environment=[env],
                                var_wifi_band=preset["data"]["wifi_band"],
                                var_num_users=var_users)
        #
        var_label_list = data_pd_y["label"].to_list()
        #
        ## load CSI amplitude
        X = load_data_x(preset["path"]["data_x"], var_label_list)


        y = encode_data_y(data_pd_y, var_task)

        if var_model == "THAT_MULTI_HEAD":
            y = reduce_dataset(y)  # CHECKKKKKKKK HEREEEEEE
        elif var_model == "THAT_ENCODER" or var_model == "DETR" or var_model== "multi_user" or var_model=="DETR_VQ" or var_model=="DETR_RVQ" or var_model == "JEPA_HYB" or var_model == "JEPA":
            y = reduce_dataset(y, preset["nn"]["num_obj_queries"])  # CHECKKKKKKKK HEREEEEEE
        elif var_model == "THAT_COUNT_CONSTRAINED":
            y_red = reduce_dataset(y)
            y = y_red.sum(axis=1)
        elif var_model == "CROWD_COUNTING_THAT":
            y = y.sum(axis = (1, 2)) # we except one number for each value of y which corresponds to count of people!

        else:
            pass

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            random_state=103)
        # np.random.randint()
        env_data_x_train.append(X_train)
        env_data_x_test.append(X_test)
        env_data_y_train.append(y_train)
        env_data_y_test.append(y_test)


    data_x_train = np.concatenate(env_data_x_train, axis = 0)
    data_x_test = np.concatenate(env_data_x_test, axis = 0)
    data_y_train = np.concatenate(env_data_y_train, axis = 0)
    data_y_test = np.concatenate(env_data_y_test, axis = 0)


    return data_x_train, data_x_test, data_y_train, data_y_test



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
    SEED = 103 
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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

    # Ensuring there is no data leakage while doing splits.
    data_train_x, data_test_x, data_train_y, data_test_y = master_splitter(preset, var_task, var_model, var_users)
    #

    # #
    ## select a WiFi-based model
    if var_model == "ST-RF": run_model = run_strf
    #
    elif var_model == "MLP": run_model = run_mlp
    #
    elif var_model == "LSTM": run_model = run_lstm
    #
    elif var_model == "CNN-1D": run_model = run_cnn_1d
    #
    elif var_model == "CNN-2D": run_model = run_cnn_2d
    #
    elif var_model == "CLSTM": run_model = run_cnn_lstm
    #
    elif var_model == "ABLSTM": run_model = run_ablstm
    #
    elif var_model == "THAT": run_model = run_that
    #
    elif var_model == "SSL": run_model = run_ssl
    #
    elif var_model == "THAT_COUNT": run_model = run_that_count_pred
    #
    elif var_model == "THAT_MULTI_HEAD": run_model = run_that_multihead
    #
    elif var_model == "THAT_COUNT_CONSTRAINED": run_model = run_that_count_pred_contrained

    elif var_model == "THAT_ENCODER": run_model = run_that_decoder

    elif var_model == "DETR": run_model = run_that_detr

    elif var_model == "CROWD_COUNTING_THAT": run_model = run_crowd_counting_THAT

    elif var_model == "multi_user": run_model = run_multi_user

    elif var_model == "JEPA_HYB": run_model =  run_JEPA_hyb

    elif var_model == "JEPA": run_model = run_JEPA

    elif var_model == "DETR_VQ": run_model = run_that_detrVQ

    elif var_model == "DETR_RVQ": run_model = run_that_detrRVQ

    else:
        raise Exception("Not valid name for model")




    #
    ## run WiFi-based model
    result = run_model(data_train_x, data_train_y,
                       data_test_x, data_test_y, var_repeat)
    #
    ##
    result["model"] = var_model
    result["task"] = var_task
    result["data"] = preset["data"]
    result["nn"] = preset["nn"]
    #
    print("\n" + "="*80)
    print(f"EXPERIMENT RESULTS - Model: {var_model}, Task: {var_task}")
    print("="*80)
    
    # Check if this is a layered result (like DETR) or single result
    if isinstance(result, dict) and any(key.startswith('layer_') for key in result.keys() if isinstance(key, str)):
        # This is a layered result (DETR models)
        print("LAYERED MODEL RESULTS:")
        for layer_key in sorted([k for k in result.keys() if isinstance(k, str) and k.startswith('layer_')]):
            layer_results = result[layer_key]
            print(f"\n{layer_key.upper()}:")
            if 'avg_precision' in layer_results:
                print(f"  Avg Precision: {layer_results['avg_precision']:.4f} ± {layer_results['se_precision']:.4f} (SE)")
            if 'avg_recall' in layer_results:
                print(f"  Avg Recall: {layer_results['avg_recall']:.4f} ± {layer_results['se_recall']:.4f} (SE)")
            if 'avg_PPP' in layer_results:
                print(f"  Avg Perfect Prediction %: {layer_results['avg_PPP']:.4f} ± {layer_results['se_PPP']:.4f} (SE)")
            if 'avg_f1_score' in layer_results:
                print(f"  Avg F1 Score: {layer_results['avg_f1_score']:.4f} ± {layer_results['se_f1_score']:.4f} (SE)")
            if 'avg_accuracy' in layer_results:
                print(f"  Avg Accuracy: {layer_results['avg_accuracy']:.4f} ± {layer_results['se_accuracy']:.4f} (SE)")
            if 'avg_total_error' in layer_results:
                print(f"  Avg Total Error: {layer_results['avg_total_error']:.4f} ± {layer_results['se_total_error']:.4f} (SE)")
    else:
        # This is a single-layer result (other models)
        print("SINGLE MODEL RESULTS:")
        if isinstance(result, dict):
            # Check for aggregated metrics first (new format)
            if 'avg_precision' in result:
                print(f"  Avg Precision: {result['avg_precision']:.4f} ± {result['se_precision']:.4f} (SE)")
            if 'avg_recall' in result:
                print(f"  Avg Recall: {result['avg_recall']:.4f} ± {result['se_recall']:.4f} (SE)")
            if 'avg_PPP' in result:
                print(f"  Avg Perfect Prediction %: {result['avg_PPP']:.4f} ± {result['se_PPP']:.4f} (SE)")
            if 'avg_f1_score' in result:
                print(f"  Avg F1 Score: {result['avg_f1_score']:.4f} ± {result['se_f1_score']:.4f} (SE)")
            if 'avg_accuracy' in result:
                print(f"  Avg Accuracy: {result['avg_accuracy']:.4f} ± {result['se_accuracy']:.4f} (SE)")
            if 'avg_total_error' in result:
                print(f"  Avg Total Error: {result['avg_total_error']:.4f} ± {result['se_total_error']:.4f} (SE)")
            
            # Fall back to single iteration metrics (old format) if aggregated metrics not available
            elif 'precision' in result:
                print(f"  Precision: {result['precision']:.4f}")
                if 'recall' in result:
                    print(f"  Recall: {result['recall']:.4f}")
                if 'perfect_prediction_percentage' in result:
                    print(f"  Perfect Prediction %: {result['perfect_prediction_percentage']:.4f}")
                if 'f1_score' in result:
                    print(f"  F1 Score: {result['f1_score']:.4f}")
                if 'accuracy' in result:
                    print(f"  Accuracy: {result['accuracy']:.4f}")
                if 'total_error' in result:
                    print(f"  Total Error: {result['total_error']:.4f}")
    
    print("\nFull Result Details:")
    print(result)
    print("="*80)
    #
    ## save results
    # var_file = open(preset["path"]["save"], 'w')
    # json.dump(result, var_file, indent=4, cls=NumpyEncoder)

#
##

if __name__ == "__main__":
    #
    ##
    run()