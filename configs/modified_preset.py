preset = {
    'wandb_name': 'd=100',
    'model': 'DETR',
    'task': 'activity',
    'repeat': 8,
    'path': {
        'data_x': '/local/data0/amir/PUBLIC_DATASET/wimans_dataset/wifi_csi/amp',
        'data_y': '/local/data0/amir/PUBLIC_DATASET/wimans_dataset/annotation.csv',
        'save': 'results/result.json'
    },
    'data': {
        'num_users': ['0', '1', '2', '3', '4', '5'],
        'wifi_band': ['5'],
        'environment': ['empty_room'],
        'length': 3000
    },
    'data_band2': {
        'num_users': ['0', '1', '2', '3', '4', '5'],
        'wifi_band': ['5'],
        'environment': ['empty_room'],
        'length': 3000
    },
    'nn': {
        'lr': 0.0005,
        'epoch': 300,
        'batch_size': 16,
        'threshold': 0.5,
        'scheduler': {
        'type': 'cosine_warmup',
        'num_warmup_epochs': 10,
        'min_lr_ratio': 0.05
    },
        'loss': {
        'type': 'HungarianMatchingLoss',
        'cost_class_weight': 1.0,
        'aux_loss_weight': 0.25,
        'label_smoothing': 0.1,
        'class_imbalance_weight': 0.25
    },
        'cross_attention_temp': 1,
        'weight_decay': 0.0002,
        'num_obj_queries': 5,
        'num_decoder_layers': 5,
        'dim_FFN': 512,
        'token_length': 100,
        'd_embedding': 10,
        'n_attention_heads': 4
    },
    'encoding': {
        'activity': {
        'nan': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'nothing': [1, 0, 0, 0, 0, 0, 0, 0, 0],
        'walk': [0, 1, 0, 0, 0, 0, 0, 0, 0],
        'rotation': [0, 0, 1, 0, 0, 0, 0, 0, 0],
        'jump': [0, 0, 0, 1, 0, 0, 0, 0, 0],
        'wave': [0, 0, 0, 0, 1, 0, 0, 0, 0],
        'lie_down': [0, 0, 0, 0, 0, 1, 0, 0, 0],
        'pick_up': [0, 0, 0, 0, 0, 0, 1, 0, 0],
        'sit_down': [0, 0, 0, 0, 0, 0, 0, 1, 0],
        'stand_up': [0, 0, 0, 0, 0, 0, 0, 0, 1]
    },
        'location': {
        'nan': [0, 0, 0, 0, 0],
        'a': [1, 0, 0, 0, 0],
        'b': [0, 1, 0, 0, 0],
        'c': [0, 0, 1, 0, 0],
        'd': [0, 0, 0, 1, 0],
        'e': [0, 0, 0, 0, 1]
    }
    },
    'pretrained_path': None,
    'transfer_scenario': 'full',
    'save_model': False,
    'saving_path': 'results/',
}
