import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np 
from Model import DOAMambaNet
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
metrics_log = []



class DOADataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path):
        self.data = torch.tensor(np.load(data_path), dtype=torch.float32)
        self.labels = torch.tensor(np.load(label_path), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def compute_rmse_with_hungarian(preds, targets):
    
    preds = np.array(preds)
    targets = np.array(targets)

    if len(preds) != len(targets):
        raise ValueError("Chiều dài của predicted và target phải bằng nhau.")

    # Tạo ma trận chi phí dựa trên sai số bình phương
    cost_matrix = (preds[:, None] - targets[None, :]) ** 2

    # Dùng thuật toán Hungarian để tìm cặp tối ưu
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Tính RMSE theo các cặp ghép tối ưu
    optimal_costs = cost_matrix[row_ind, col_ind]
    rmse = np.sqrt(optimal_costs.mean())
    return rmse

def train_model(train_data_path, train_label_path, test_data_path, test_label_path, model_save_path='doa_mamba_model_multi_100.pth', num_epochs=50, batch_size=64, lr=1e-3):
    if not os.path.exists(train_data_path) or not os.path.exists(train_label_path):
        print("Không tìm thấy file dữ liệu huấn luyện.")
        return
    if not os.path.exists(test_data_path) or not os.path.exists(test_label_path):
        print("Không tìm thấy file dữ liệu kiểm tra.")
        return

    train_dataset = TensorDataset(torch.tensor(np.load(train_data_path), dtype=torch.float32), torch.tensor(np.load(train_label_path), dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(np.load(test_data_path), dtype=torch.float32), torch.tensor(np.load(test_label_path), dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DOAMambaNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    accs = []
    test_losses = []
    test_accs = []

    for epoch in range(num_epochs):
    # ------- Training -------
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_labels = 0
        total_samples = 0
        exact_match = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = torch.zeros_like(labels)
            topk_values = labels.sum(dim=1).int()

            for i in range(labels.size(0)):
                k_i = topk_values[i].item()
                if k_i > 0:
                    topk_idx = torch.topk(probs[i], k=k_i).indices
                    preds[i, topk_idx] = 1

            labels_int = labels.int()
            total_correct += (preds == labels_int).sum().item()
            total_labels += labels.numel()
            exact_match += ((preds == labels_int).all(dim=1)).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = exact_match / total_samples

        # ------- Evaluation -------
        model.eval()
        with torch.no_grad():
            test_total_loss = 0.0
            test_total_samples = 0
            test_exact_match = 0

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = torch.zeros_like(labels)
                topk_values = labels.sum(dim=1).int()

                for i in range(labels.size(0)):
                    k_i = topk_values[i].item()
                    if k_i > 0:
                        topk_idx = torch.topk(probs[i], k=k_i).indices
                        preds[i, topk_idx] = 1

                test_exact_match += ((preds == labels.int()).all(dim=1)).sum().item()
                test_total_samples += labels.size(0)

            avg_test_loss = test_total_loss / len(test_loader)
            test_acc = test_exact_match / test_total_samples
            
        losses.append(avg_loss)
        accs.append(train_acc)
        test_losses.append(avg_test_loss)
        test_accs.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}")

        metrics_log.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_acc': train_acc,
                'test_loss': avg_test_loss,
                'test_acc': test_acc,
            })

    torch.save(model.state_dict(), model_save_path)
    model.eval()
    all_rmse = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.numpy()

            true_angles = [np.where(l == 1)[0] - 90 for l in labels]
            pred_angles = []
            for i, p in enumerate(probs):
                k = int(np.sum(labels[i]))
                topk_idx = np.argsort(p)[-k:]
                pred_angles.append(topk_idx - 90)

            for t, p in zip(true_angles, pred_angles):
                if len(t) == len(p) and len(t) > 0:
                    rmse = compute_rmse_with_hungarian(p, t)
                    all_rmse.append(rmse)
    avg_rmse = np.mean(all_rmse) if all_rmse else None
    if metrics_log:
        metrics_log[-1]['rmse'] = avg_rmse
    if all_rmse:
        print(f"✅ RMSE trung bình trên tập test: {np.mean(all_rmse):.2f}°")
    else:
        print("⚠️ Không thể tính RMSE: không có mẫu hợp lệ.")


if __name__ == "__main__":
    train_model(
        'doa_train_signals_mamba_multiSNR.npy',
        'doa_train_labels_mamba_multiSNR.npy',
        'doa_test_signals_mamba_multiSNR.npy',
        'doa_test_labels_mamba_multiSNR.npy'
    )

df = pd.DataFrame(metrics_log)
df.to_csv("MAMBA_100.csv", index=False)