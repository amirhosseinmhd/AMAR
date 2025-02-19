import os
"""
[file]          preset.py
[description]   default settings of WiFi-based models
"""
#
##
preset = {
    #
    ## define model
    "model": "DETR",                                    # "ST-RF", "MLP", "LSTM", "CNN-1D", "CNN-2D", "CLSTM", "ABLSTM", "THAT",
                                                              # "THAT_COUNT", "THAT_ENCODER", THAT_COUNT_CONSTRAINED, THAT_MULTI_HEAD DETR
                                                            #JOINT_DETR
    # "model": "MLP",
    ## define task
    "task": "activity",                                 # "identity", "activity", "location"
    #
    ## number of repeated experiments
    "repeat": 5,
    ## path of data
    "path": {
        "data_x": "/local/data0/amir/PUBLIC_DATASET/wimans_dataset/wifi_csi/amp",               # directory of CSI amplitude files
        "data_y": "/local/data0/amir/PUBLIC_DATASET/wimans_dataset/annotation.csv",             # path of annotation file
        "save": "results/result.json"                           # path to save results
    },
    #
    ## data selection for experiments
    "data": {
        "num_users": ["0","1", "2", "3", "4", "5"] ,   # select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
        "wifi_band": ["5"],                           # select WiFi band(s) (e.g., ["2.4"], ["5"], ["2.4", "5"])
        "environment": ["empty_room"],                   # select environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
        "length": 3000,                                 # default length of CSI
    },
    "data_band2": {
        "num_users": ["0","1", "2", "3", "4", "5"] ,  # select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
        "wifi_band": ["5"],  # select WiFi band(s) (e.g., ["2.4"], ["5"], ["2.4", "5"])
        "environment": ["empty_room"],  # select environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
        "length": 3000,  # default length of CSI
    }
    ,
    #
    ## hyperparameters of models
    "nn": {
        "lr": 5e-4,                                     # learning rate
        "epoch": 300,                                   # number of epochs
        "batch_size": 16,                              # batch size
        "threshold": 0.5,                               # threshold to binarize sigmoid outputs
        "scheduler": {
            "type": "cosine_warmup",  # type of scheduler
            "num_warmup_epochs": 10,  # number of warmup epochs
            "min_lr_ratio": 0.05  # minimum learning rate ratio
        },
        # Loss function parameters
        "loss": {
            "type": "HungarianMatchingLoss",  # type of loss function
            "cost_class_weight": 1.0,  # weight for classification cost
            "aux_loss_weight": 0.25,  # weight for auxiliary losses
            "label_smoothing": 0.1,  # label smoothing factor
            "class_imbalance_weight": 0.25
        },
        "cross_attention_temp": 1,
        "weight_decay": 2e-4,
        "num_obj_queries": 5,
        "num_decoder_layers": 5,
        "dim_FFN": 512,
        "token_length": 200,
        "d_embedding": 16
    },
    #
    ## encoding of activities and locations
    "encoding": {
        "activity": {                                   # encoding of different activities
            "nan":      [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "nothing":  [1, 0, 0, 0, 0, 0, 0, 0, 0],
            "walk":     [0, 1, 0, 0, 0, 0, 0, 0, 0],
            "rotation": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "jump":     [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "wave":     [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "lie_down": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "pick_up":  [0, 0, 0, 0, 0, 0, 1, 0, 0],
            "sit_down": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "stand_up": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        },        # "location": {
        #     "nan": [0, 0],  # Default origin point
        #     "a": [0.80, 4.45],  # Location A coordinates
        #     "b": [0.80, 2.30],  # Location B coordinates
        #     "c": [3.60, 4.45],  # Location C coordinates
        #     "d": [3.60, 1.90],  # Location D coordinates
        #     "e": [8.30, 4.45],  # Location E coordinates
        # }
        "location": {                                   # encoding of different locations
            "nan":  [0, 0, 0, 0, 0],
            "a":    [1, 0, 0, 0, 0],
            "b":    [0, 1, 0, 0, 0],
            "c":    [0, 0, 1, 0, 0],
            "d":    [0, 0, 0, 1, 0],
            "e":    [0, 0, 0, 0, 1],
        },
    },
    # "pretrained_path": "/home/amirmhd/Documents/multi_modal_CSI/benchmark/wifi_csi/results/model_0/PT_classroom_meeting_room_DETR.pth",  # Path to pretrained models
    "pretrained_path": None,
    "transfer_scenario": "full",  # One of ["full", "feature_extractor", "feature_encoder"]
    "save_model": False,  # Whether to save model components
    "saving_path": "results/"
}
