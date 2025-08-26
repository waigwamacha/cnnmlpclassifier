
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch

from datetime import datetime
from pyprojroot import here

hostname = os.uname()[1]
proj_dir = here()
datem = datetime.today().strftime("%Y-%m-%d")
# Learning rate scheduler (exponentially increasing LR)
#def lr_lambda(epoch):
#    return 10 ** (epoch / 10)  # Increases LR exponentially

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


def create_three_year_bins(df: pd.DataFrame):
    bins = list(range(3, 31, 3)) + [31]  # [5, 8, 11, ..., 29, 31]
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    df['age_bracket'] = pd.cut(df['scan_age'], bins=bins, labels=labels, right=False, include_lowest=True)
    df['age_bracket_class'] = df['age_bracket'].cat.codes
    return df

def training_loss_visualization(tracking_loss, trackingval_loss, tracking_mean_trainloss, tracking_mean_valloss):

    plt.figure(figsize=(12, 10))  
    plt.plot(range(1, len(tracking_loss) + 1), tracking_loss, label="Training loss")
    plt.plot(range(1, len(trackingval_loss) + 1), trackingval_loss, label="Validation loss")
    plt.plot(range(1, len(tracking_mean_trainloss) + 1), tracking_mean_trainloss, label="Baseline train loss")
    plt.plot(range(1, len(tracking_mean_valloss) + 1), tracking_mean_valloss, label="Baseline validation loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{proj_dir}/figures/{hostname}-{datem}-trainingvalidationloss.png", dpi=300)

