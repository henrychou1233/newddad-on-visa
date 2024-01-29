import torch
from torchmetrics import AUROC
import os
from skimage import measure
import pandas as pd
from statistics import mean
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
import time
from sklearn import metrics

def metric(labels_list, predictions, anomaly_map_list, gt_list, config):
    start_time = time.time()  # 开始计时

    labels_list = torch.tensor(labels_list)
    predictions = torch.tensor(predictions)
    pro = compute_pro(gt_list, anomaly_map_list, num_th=200)

    results_embeddings = anomaly_map_list[0]
    for feature in anomaly_map_list[1:]:
        results_embeddings = torch.cat((results_embeddings, feature), 0)
    results_embeddings = ((results_embeddings - results_embeddings.min()) / 
                          (results_embeddings.max() - results_embeddings.min()))

    gt_embeddings = gt_list[0]
    for feature in gt_list[1:]:
        gt_embeddings = torch.cat((gt_embeddings, feature), 0)

    results_embeddings = results_embeddings.clone().detach().requires_grad_(False)
    gt_embeddings = gt_embeddings.clone().detach().requires_grad_(False)

    auroc = AUROC(task="binary")
    auroc_score = auroc(predictions, labels_list)

    gt_embeddings = torch.flatten(gt_embeddings).type(torch.bool).cpu().detach()
    results_embeddings = torch.flatten(results_embeddings).cpu().detach()

    auroc_pixel = auroc(results_embeddings, gt_embeddings)

    r_gt_embeddings = gt_embeddings.cpu().detach().numpy().ravel()
    r_results_embeddings = results_embeddings.cpu().detach().numpy().ravel()
    precision, recall, thresholds = metrics.precision_recall_curve(
        r_gt_embeddings.astype(int), r_results_embeddings
    )

    if len(thresholds) > 0:
        F1_scores = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )
        thresholdOpt = thresholds[np.argmax(F1_scores)]
    else:
        thresholdOpt = 0.5  # 设置默认阈值
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    thresholdOpt = thresholds[np.argmax(F1_scores)]

    if config.metrics.image_level_AUROC:
        print(f'AUROC: {auroc_score}')
    if config.metrics.pixel_level_AUROC:
        print(f"AUROC pixel level: {auroc_pixel}")
    if config.metrics.pro:
        print(f'PRO: {pro}')
    
    max_f1 = np.max(F1_scores)
    print(f'Max F1 Score: {max_f1}')

    end_time = time.time()  # 结束计时
    total_time = end_time - start_time
    num_frames = len(labels_list)
    fps = num_frames / total_time if total_time > 0 else 0
    latency = total_time * 1000  # 转换为毫秒
    print(f"FPS: {fps}")
    print(f"Latency: {latency} ms")
    return thresholdOpt, max_f1, fps, latency



#https://github.com/hq-deng/RD4AD/blob/main/test.py#L337
def compute_pro(masks, amaps, num_th = 200):
    resutls_embeddings = amaps[0]

    for feature in amaps[1:]:
        resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
    print(resutls_embeddings,"------------------------------------------")
    amaps =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 
    amaps = amaps.squeeze(1)
    amaps = amaps.cpu().detach().numpy()
    print(amaps,"::::::::::::::::amaps::::::::::::::::::::")
    gt_embeddings = masks[0]
    print(masks[0],"#############masks[0]################") #問題出在這
    
    for feature in masks[1:]:
        gt_embeddings = torch.cat((gt_embeddings, feature), 0)
    masks = gt_embeddings.squeeze(1).cpu().detach().numpy()
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1
        print(binary_amaps,"**********binary_amaps**********")
        print(masks,"+++++++++masks++++++++++++")
        pros = []
        print(pros,"---------------this is pros----------------")
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                print("--------------enter this loop---------------")
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                print("--------axes0_ids---------",axes0_ids)
                print("-------------axes1_ids---------",axes1_ids)
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                # print(len(tp_pixels),"************tp_pixels**************") 這是錯的因為他沒有長度
                # print("************ region.area ************",len(region.area))
                if region.area > 0:
                    pros.append(tp_pixels / region.area)
                print(pros,"---------------this is pros after----------------")

        print("************pros length:************", len(pros))
        # 檢查是否至少有一個數據點
        if len(pros) > 0:
            pro_mean = mean(pros)
        else:
            pro_mean = 0.0

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks , binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        print(f"Threshold: {th}, FPR: {fpr}, PRO: {mean(pros)}")
        print(pros,fpr,th,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        df = pd.concat([df, pd.DataFrame({"pro": mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)

        # df = df.concat({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc