import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
import torch.nn as nn



# def validate(model, opt):
#     #------------------------------------自己加的
#     device = next(model.parameters()).device
#     #=======================================
#     data_loader = create_dataloader(opt)
#
#     with torch.no_grad():
#         y_true, y_pred = [], []
#         # for img, label in data_loader:
#         #     in_tens = img.cuda()
#         #     y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
#         #     y_true.extend(label.flatten().tolist())
#         #---------------------------------------自己加的
#         for img, label in data_loader:
#             img = img.to(device)
#             label = label.to(device)
#
#             y_pred.extend(
#                 model(img).sigmoid().flatten().tolist()
#             )
#             y_true.extend(
#                 label.flatten().tolist()
#             )
#         #===================================================
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
#     f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
#     acc = accuracy_score(y_true, y_pred > 0.5)
#     ap = average_precision_score(y_true, y_pred)
#     return acc, ap, r_acc, f_acc, y_true, y_pred

def validate(model, opt):
    device = next(model.parameters()).device
    model.eval()

    dataloader = create_dataloader(opt)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    count = 0

    with torch.no_grad():
        for data in dataloader:

            # -------- CodeBERT --------
            if isinstance(data, dict):
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                labels = data["label"].to(device).float()

                outputs = model(input_ids, attention_mask).squeeze()

            # -------- CNN / MLP --------
            else:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                outputs = model(inputs).squeeze()

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            count += 1

            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / count
    val_acc = accuracy_score(all_labels, all_preds)

    return val_loss, val_acc

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
