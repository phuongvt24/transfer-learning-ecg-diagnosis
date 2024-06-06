import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from ptbxl.ptbxl_dataset import PTBXL  # Điều chỉnh theo cấu trúc thư mục của bạn
from CDIL import CDILClassifier  # Điều chỉnh theo mô hình bạn sử dụng
from eval import classify, get_f1, get_auprc
from resnet50 import ResNet1d50
from resnet1d18 import ResNet1d18
from lstm import LSTM, BiLSTM
from gru import GRU_Classifier
from eff import EfficientNetB0

# def load_model(model_class, checkpoint_path, device):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model = model_class(num_classes=len(PTBXL.CLASSES))
#     model.load_state_dict(checkpoint['state_dict'])
#     model.to(device)
#     return model

# def load_model(model_class, checkpoint_path, device, **kwargs):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model = model_class(device=device, **kwargs)
#     model.load_state_dict(checkpoint['state_dict'])
#     model.to(device)
#     return model

def load_model(model_class, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_class(classes=len(PTBXL.CLASSES))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model

def get_precision(y_trues, y_preds):
    precision = []
    for j in range(y_trues.shape[1]):
        precision.append(precision_score(y_trues[:, j], y_preds[:, j]))
    return np.array(precision)

def get_recall(y_trues, y_preds):
    recall = []
    for j in range(y_trues.shape[1]):
        recall.append(recall_score(y_trues[:, j], y_preds[:, j]))
    return np.array(recall)

def main():
    # Thông số
    model_checkpoint = './ptbxl/checkpoints/EfficientNetB0_ptbxl_2024-06-05T12:59:52.351058.pt'  # Thay bằng đường dẫn tệp lưu mô hình của bạn
    data_dir = 'ptbxl'
    batch_size = 128

    # Kiểm tra thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tạo tập dữ liệu test
    test_data = PTBXL('test', f'{data_dir}/data/')

    # Tải mô hình đã lưu
    model = load_model(EfficientNetB0, model_checkpoint, device)
    # model = load_model(CDILClassifier, model_checkpoint, device, num_classes=len(PTBXL.CLASSES))


    # Đánh giá mô hình trên tập test
    y_trues, y_preds = classify(model, device, test_data, 1)

# Lấy tên các class
    class_names = PTBXL.CLASSES

    # Tính F1 scores
    f1_scores = get_f1(y_trues, y_preds)

    # Tính Precision
    precision_scores = get_precision(y_trues, y_preds)

    # Tính Recall
    recall_scores = get_recall(y_trues, y_preds)

    # Tính AUC
    aucs = get_auprc(y_trues, y_preds)

    # Tính Support
    support = np.sum(y_trues, axis=0)

    # Tạo báo cáo chi tiết
    print("Detailed classification report:")
    for idx, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {precision_scores[idx]}")
        print(f"  Recall: {recall_scores[idx]}")
        print(f"  F1 Score: {f1_scores[idx]}")
        print(f"  AUC: {aucs[idx]}")
        print(f"  Support: {support[idx]}")
        print()
    print(class_names)

    # Tạo báo cáo tổng hợp bằng classification_report
    y_trues_flat = np.argmax(y_trues, axis=1)
    y_preds_flat = np.argmax(y_preds, axis=1)
    report = classification_report(y_trues_flat, y_preds_flat, target_names=class_names)
    print(report)

if __name__ == '__main__':
    main()
