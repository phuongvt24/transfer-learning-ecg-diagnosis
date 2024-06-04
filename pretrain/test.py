import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from ptbxl.ptbxl_dataset import PTBXL  # Điều chỉnh theo cấu trúc thư mục của bạn
from CDIL import CDILClassifier  # Điều chỉnh theo mô hình bạn sử dụng
from eval import classify, get_f1, get_auprc
from resnet50 import ResNet1d50
from resnet1d18 import ResNet1d18
from lstm import LSTM, BiLSTM
from gru import GRU_Classifier

def load_model(model_class, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_class(num_classes=len(PTBXL.CLASSES))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model

def main():
    # Thông số
    model_checkpoint = './ptbxl/checkpoints/resnet1d18_ptbxl_2024-05-02T05_10_46.358852.pt'  # Thay bằng đường dẫn tệp lưu mô hình của bạn
    data_dir = 'ptbxl'
    batch_size = 128

    # Kiểm tra thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tạo tập dữ liệu test
    test_data = PTBXL('test', f'{data_dir}/data/')

    # Tải mô hình đã lưu
    model = load_model(ResNet1d18, model_checkpoint, device)

    # Đánh giá mô hình trên tập test
    y_trues, y_preds = classify(model, device, test_data, 1)

    # Tính F1 scores
    f1_scores = get_f1(y_trues, y_preds)
    print("F1 scores:", f1_scores)
    print("Mean F1 score:", f1_scores.mean())

    # Tính Precision, Recall
    precision = precision_score(y_trues, y_preds, average='macro')
    recall = recall_score(y_trues, y_preds, average='macro')
    print("Precision:", precision)
    print("Recall:", recall)

    # Tính AUC
    # Chuyển y_trues và y_preds thành dạng nhị phân (1 lớp cho mỗi lần tính)
    aucs = get_auprc(y_trues, y_preds)
    print("AUC scores:", aucs)
    print("Mean AUC score:", aucs.mean())

if __name__ == '__main__':
    main()
