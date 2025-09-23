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
    "jepa_pretrained_path": "/Users/amirmhd/Downloads/best_model.pth",
    "finetune_strategy": "finetune_encoder_small_lr",
    "wandb_name": "00",
    "model": "DETR_RVQ",                                    # "ST-RF", "MLP", "LSTM", "CNN-1D", "CNN-2D", "CLSTM", "ABLSTM", "THAT",
                                 # "ST-RF", "MLP", "LSTM", "CNN-1D", "CNN-2D", "CLSTM", "ABLSTM", "THAT",
                                                              # "THAT_COUNT", "THAT_ENCODER", THAT_COUNT_CONSTRAINED, THAT_MULTI_HEAD DETR
                                                            #JOINT_DETR, JEPA_HYB, "JEPA", "DETR_VQ", "DETR_RVQ"
    # "model": "MLP",
    ## define task
    "task": "activity",                                 # "identity", "activity", "location"
    #
    ## number of repeated experiments
    "repeat": 8,
    ## path of data
    "path": {

        "data_x": "/home/amirmhd/Documents/multi_modal_CSI/dataset/wifi_csi/amp",
        "data_y": "/home/amirmhd/Documents/multi_modal_CSI/dataset/annotation.csv",
        # "data_x": "/local/data0/amir/PUBLIC_DATASET/wimans_dataset/wifi_csi/amp",  # directory of CSI amplitude files
        # "data_y": "/local/data0/amir/PUBLIC_DATASET/wimans_dataset/annotation.csv",  # path of annotation file
        # "data_x": "/Users/amirmhd/Documents/MASc/Research/Data/Wimans/wifi_csi/amp",  # directory of CSI amplitude files
        # "data_y": "/Users/amirmhd/Documents/MASc/Research/Data/Wimans/annotation.csv",  # path of annotation file
        # "data_x": "/home/amirmhd/projects/def-hinat/amirmhd/Dataset/wifi_csi/amp",  # directory of CSI amplitude files
        # "data_y": "/home/amirmhd/projects/def-hinat/amirmhd/Dataset/annotation.csv",  # path of annotation file
        "save": "results/result.json"                           # path to save results
    },
    #
    ## data selection for experiments
    "data": {
        "num_users": ["0","1", "2", "3", "4", "5"] ,   # select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
        "wifi_band": ["5"],                           # select WiFi band(s) (e.g., ["2.4"], ["5"], ["2.4", "5"])
        "environment": ["meeting_room"],                   # select environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
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
        "epoch": 300,                                   # number of epochs
        "batch_size":16,                              # batch size
        "threshold": 0.5,                               # threshold to binarize sigmoid outputs
        "scheduler": {
            "type": "cosine_warmup",  # type of scheduler
            "num_warmup_epochs": 3,  # number of warmup epochs
            "min_lr_ratio": 0.1  # minimum learning rate ratio
        },
        # Loss function parameters
        "loss": {
            "SSL_coeff": 0.2 ,
            "type": "HungarianMatchingLoss",  # type of loss function
            "cost_class_weight": 1.0,  # weight for classification cost
            "aux_loss_weight": 0.25,  # weight for auxiliary losses
            "label_smoothing": 0.05,  # label smoothing factor
            "class_imbalance_weight": 0.25
        },
        "KMEANS_Initialization": False,
        "cross_attention_temp": 1,
        "weight_decay": 1e-4,
        "num_obj_queries": 5,
        "num_decoder_layers":6,
        "dim_FFN": 512,
        "token_length": 188, #74
        "d_embedding": 64,
        "n_layers_encoder":4,
        "n_attention_heads": 4,
        "query_dropout_rate": 0.0,
        "commitment_cost": 0.25,
        "num_codes": 64,
        "num_rvq_layers":4,  # Number of RVQ layers (V+1 in the formula)
        "quantization_dropout": 0.2,  # Dropout rate for quantization layers
        "PCA": False

},
    "jepa": {
        "segment_length": 48,              # Number of timestamps in each segment.
        "num_segments_total_view": 1,      # Total number of segments considered in a single processing view.
        "encoder_layers": 6,                # Number of layers in the Transformer Encoder.
        "ema_decay": 0.998,                 # Decay rate for the Exponential Moving Average of target encoder weights.
        "num_target_blocks":4,             # Number of target blocks to sample and predict.
        "target_block_size_segments": 20,    # Number of contiguous segments forming a single target block.

        # --- Predictor Specific Configurations ---
        # The "narrow" dimension of the predictor's internal Transformer.
        "predictor_d_model": 52,
        "predictor_attention_heads": 4,       # Number of attention heads in the Predictor's Transformer.
        "predictor_layers": 3,                # Number of layers in the Predictor's Transformer.
        "sampling_weight_decay_factor": 0.9,  # Factor to reduce weights after sampling
        "sampling_weight_reset_interval": 100,  # Reset weights every N times
        "log_sampling_stats_interval": 100,  # Log sampling statistics every N batches
        "loss": {
            "prediction_coef": 1.0,  # Keep JEPA prediction loss at full strength
            "vicreg_coef": 0.1,  # Much smaller - VICReg is regularization, not main loss
            "vicreg_sim_coef": 25.0,  # NEW: Similarity between different blocks
            "vicreg_std_coef": 25.0,  # Variance regularization
            "vicreg_cov_coef": 1.0  # Covariance regularization
        }
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
        # "location": {
        #     "nan": [0, 0],  # Default origin point
        #     "a": [3.700, 2.65],  # Location A coordinates
        #     "b": [3.700, 6.250],  # Location B coordinates
        #     "c": [2.500, 4.450],  # Location C coordinates
        #     "d": [1.400, 2.650],  # Location D coordinates
        #     "e": [1.400, 6.250],  # Location E coordinates
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
    "pretrained_path": None,
    # "pretrained_path": "/saved_models/jepa_ssl_empty_room+classroom+meeting_room_20250721_124829/jepa_ssl_final_empty_room+classroom+meeting_room_20250721_124829.pth",
    "transfer_scenario": "freeze_encoder",  # One of ["full", "feature_extractor", "feature_encoder"]
    "save_model": False,  # Whether to save model components
    "saving_path": "/home/amirmhd/Documents/multi_modal_CSI/benchmark/wifi_csi/results/"
}


preset["wandb_name"] = "N_LAYERS_" + str(preset["nn"]["num_rvq_layers"]) + "Num_Codes_" + str(preset["nn"]["num_codes"])
