import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from src.data.load_data import load_data_x, load_data_y, encode_data_y
from configs.preset import preset
from model.SSL_model import SS_Model, InferenceDataset
import json


def load_model(model_path, var_x_shape, var_y_shape, device):
    model = SS_Model(var_x_shape, var_y_shape).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Remove the '_orig_mod.' prefix from keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def evaluate_model(model, test_loader, var_threshold, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            predict_test_y = model(x, inference=True)
            predict_test_y = (torch.sigmoid(predict_test_y) > var_threshold).float()
            all_preds.append(predict_test_y.cpu())
            all_labels.append(labels.cpu())

    predict_test_y = torch.cat(all_preds, dim=0).numpy()
    data_test_y = torch.cat(all_labels, dim=0).numpy()

    return predict_test_y, data_test_y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment=preset["data"]["environment"],
                            var_wifi_band=preset["data"]["wifi_band"],
                            var_num_users=preset["data"]["num_users"])

    var_label_list = data_pd_y["label"].to_list()
    data_x = load_data_x(preset["path"]["data_x"], var_label_list)
    data_y = encode_data_y(data_pd_y, preset["task"])

    # Reshape data
    data_x = data_x.reshape(data_x.shape[0], data_x.shape[1], -1)
    var_x_shape, var_y_shape = data_x[0].shape, data_y[0].reshape(-1).shape

    # Create test dataset and dataloader
    test_dataset = InferenceDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))
    test_loader = DataLoader(test_dataset, batch_size=preset["nn"]["batch_size"], shuffle=False)

    # Load the model
    model_path = "best_model_r0.pth"  # Adjust this path as needed
    model = load_model(model_path, var_x_shape, var_y_shape, device)

    # Evaluate the model
    predict_test_y, data_test_y = evaluate_model(model, test_loader, preset["nn"]["threshold"], device)

    # Calculate accuracy
    data_test_y_c = data_test_y.reshape(-1, data_test_y.shape[-1])
    predict_test_y_c = predict_test_y.reshape(-1, data_test_y.shape[-1])
    accuracy = accuracy_score(data_test_y_c.astype(int), predict_test_y_c.astype(int))

    # Generate classification report
    report = classification_report(data_test_y_c, predict_test_y_c, digits=6, zero_division=0, output_dict=True)

    # Print results
    print(f"Model Accuracy: {accuracy:.6f}")
    print("\nClassification Report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()