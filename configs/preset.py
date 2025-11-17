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
    "model": "AMAR_WO_RVQ",                                    #"ABLSTM", "BCE_THAT", ABLSTM_COUNT
                                                              # "DEM_THAT",  AMAR, AMAR_WO_RVQ
                                                            # 
    # "model": "MLP",
    ## define task
    "task": "activity",                                 #  "activity"
    #
    ## number of repeated experiments
    "repeat": 2,
    ## path of data
    "path": {
        "data_x": "/local/data0/amir/PUBLIC_DATASET/wimans_dataset/wifi_csi/amp",  # directory of CSI amplitude files
        "data_y": "/local/data0/amir/PUBLIC_DATASET/wimans_dataset/annotation.csv",  # path of annotation file
        "save": "results/result.json"                           # path to save results
    },
    #
    ## data selection for experiments
    "data": {
        "num_users": ["0","1", "2", "3", "4", "5"] ,   # select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
        "wifi_band": ["5"],                           # select WiFi band(s) (e.g., ["2.4"], ["5"], ["2.4", "5"])
        "environment": ["empty_room"],               # select environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
        "length": 3000,                                 # default length of CSI
    },
    "data_band2": {
        "num_users": ["0","1", "2", "3", "4", "5"] ,  # select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
        "wifi_band": ["5"],  # select WiFi band(s) (e.g., ["2.4"], ["5"], ["2.4", "5"])
        "environment": ["classroom"],  # select environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
        "length": 3000,  # default length of CSI
    }
    ,
    #
    ## hyperparameters of models
    "nn": {
        "lr": 5e-4,                                     # learning rate
        "epoch": 1,                                   # number of epochs
        "batch_size":16,                              # batch size
        "threshold": 0.5,                               # threshold to binarize sigmoid outputs
        "scheduler": {
            "type": "cosine_warmup",  # type of scheduler
            "num_warmup_epochs": 3,  # number of warmup epochs
            "min_lr_ratio": 0.1  # minimum learning rate ratio
        },
        # Loss function parameters
        "loss": {
            "cost_class_weight": 1.0,  # weight for classification cost
            "aux_loss_weight": 0.25,  # weight for auxiliary losses
            "label_smoothing": 0.2,  # label smoothing factor
            "class_imbalance_weight": 0.25
        },
        "cross_attention_temp": 1,
        "weight_decay": 1e-4,
        "num_obj_queries": 6,
        "num_decoder_layers":6,
        "dim_FFN": 512,
        "token_length": 188, #74
        "d_embedding": 64,
        "n_layers_encoder":4,
        "n_attention_heads": 4,
        "query_dropout_rate": 0,
        "commitment_cost": 0.5,
        "num_codes": 16,
        "num_rvq_layers":4,  # Number of RVQ layers 
        "quantization_dropout": 0.3,  # Dropout rate for quantization layers

},
    

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
        },
        "location": {                                   # encoding of different locations
            "nan":  [0, 0, 0, 0, 0],
            "a":    [1, 0, 0, 0, 0],
            "b":    [0, 1, 0, 0, 0],
            "c":    [0, 0, 1, 0, 0],
            "d":    [0, 0, 0, 1, 0],
            "e":    [0, 0, 0, 0, 1],
        },
    },
    "pretrained_path": None,
    # "pretrained_path": "/saved_models/jepa_ssl_empty_room+classroom+meeting_room_20250721_124829/jepa_ssl_final_empty_room+classroom+meeting_room_20250721_124829.pth",
    "transfer_scenario": "freeze_encoder",  # One of ["full", "feature_extractor", "feature_encoder"]
    "save_model": False,  # Whether to save model components
    "saving_path": "/home/amirmhd/Documents/multi_modal_CSI/results/checkpoints/"
}


preset["wandb_name"] = "N_LAYERS_" + str(preset["nn"]["num_rvq_layers"]) + "Num_Codes_" + str(preset["nn"]["num_codes"])
