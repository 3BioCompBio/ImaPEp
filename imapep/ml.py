import json
import shutil
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from imapep.utils import *
from sklearn.model_selection import KFold, StratifiedKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


class CNN(nn.Module):
    def __init__(self, conv_in, conv_out, fc_dim, dropout, pool_size=4, bias=True):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(conv_in, conv_out, (1, 1), bias=bias)
        self.resnet1_layers = nn.ModuleList([
            nn.Conv2d(conv_out, conv_out, (3, 3), padding="same", bias=bias),
            nn.Conv2d(conv_out, conv_out, (3, 3), padding="same", bias=bias)
        ])
        self.resnet2_layers = nn.ModuleList([
            nn.Conv2d(conv_out, conv_out, (3, 3), padding="same", bias=bias),
            nn.Conv2d(conv_out, conv_out, (3, 3), padding="same", bias=bias)
        ])
        self.maxpool = nn.MaxPool2d(pool_size)
        self.avgpool = nn.AvgPool2d(pool_size)
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_dim, 1, bias=bias)
        self.relu = nn.LeakyReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def resnet1(self, x):
        y = self.resnet1_layers[0](x)
        y = self.relu(y)
        y = self.resnet1_layers[1](y)
        return self.relu(x + y)

    def resnet2(self, x):
        y = self.resnet2_layers[0](x)
        y = self.relu(y)
        y = self.resnet2_layers[1](y)
        return self.relu(x + y)
    
    def forward(self, x: Tensor) -> Tensor:
        """ x should be 4-dimensional torch.Tensor with
            (N, W, H, C_in) format. 
        """
        x = self.conv1(x)
        x = self.resnet1(x)
        x = self.resnet2(x)
        x_max = self.maxpool(x)
        x_avg = self.avgpool(x)
        x = torch.cat([x_max, x_avg], dim=1)
        x = self.dropout(x)
        x = self.flat(x)
        logits = self.fc(x)
        scores = torch.sigmoid(logits)
        return scores


class ProteinDataset(Dataset):
    def __init__(self, data_dir, device, shape=100, mode="OOO"):
        super(ProteinDataset, self).__init__()
        self.data_dir = Path(data_dir)
        self.device = device
        self.shape = shape
        self.mode = mode
        with open(self.data_dir/"id_to_target.json", "r") as fp:
            id_to_label = json.load(fp)
        self.ids = list(id_to_label.keys())
        self.targets = list(id_to_label.values())


    @staticmethod
    def get_sample_type(id_):
        if "+" not in id_ and "@" not in id_:
            return 1
        elif "+" in id_:
            return 2
        elif "@" in id_:
            if "rot" in id_:
                return 3
            elif "trl" in id_:
                return 4

    @staticmethod    
    def _get_separate_ids(id_):
        """Get antibody and antigen IDs from a complex ID. """
        if "+" not in id_ and "@" not in id_:
            return id_, id_
        elif "+" in id_:
            return id_.split("+")[0], id_.split("+")[1]
        elif "@" in id_:
            return id_.split("$")[0], id_.split("$")[0]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        id_ = self.ids[k]
        label = self.targets[k]
        
        if "+" not in id_ and "@" not in id_:  # positive sample
            ab_img_fpath = self.data_dir / f"{id_}-ab.png"
            T_ab = read_image(str(ab_img_fpath))/255
            ag_img_fpath = self.data_dir / f"{id_}-ag.png"
            T_ag = read_image(str(ag_img_fpath))/255
            T = torch.cat([T_ab, T_ag], dim=0)
        elif "+" in id_:  # shuffled negative
            id_ab = id_.split("+")[0]
            id_ag = id_.split("+")[1]
            ab_img_fpath = self.data_dir / f"{id_ab}-ab.png"
            T_ab = read_image(str(ab_img_fpath))/255
            ag_img_fpath = self.data_dir / f"{id_ag}-ag.png"
            T_ag = read_image(str(ag_img_fpath))/255
            T = torch.cat([T_ab, T_ag], dim=0)
        elif "@" in id_:
            suffix = id_.split("@")[1]
            id_prefix = id_.split("$")[0]
            if suffix == "ag":
                ab_img_fpath = self.data_dir / f"{id_prefix}-ab.png"
                ag_img_fpath = self.data_dir / f"{id_}.png"
            elif suffix == "ab":  # Operation was on Ab
                ag_img_fpath = self.data_dir / f"{id_prefix}-ag.png"
                ab_img_fpath = self.data_dir / f"{id_}.png"
            T_ab = read_image(str(ab_img_fpath))/255
            T_ag = read_image(str(ag_img_fpath))/255
            T = torch.cat([T_ab, T_ag], dim=0)
        
        crop = (200 - self.shape) // 2
        T_ab = T[:3,crop:-crop,crop:-crop]
        T_ag = T[3:,crop:-crop,crop:-crop]

        if self.mode != "OOO":
            ab_img_channels = []
            ag_img_channels = []
            for i in range(len(self.mode)):
                x = self.mode[i]
                if x == "O":
                    ab_img_channel = T_ab[[i],...]
                    ag_img_channel = T_ag[[i],...]
                elif x == "X":
                    ab_img_channel = torch.zeros_like(T_ab[[i],...])
                    ag_img_channel = torch.zeros_like(T_ag[[i],...])
                ab_img_channels.append(ab_img_channel)
                ag_img_channels.append(ag_img_channel)
            T_ab = torch.cat(ab_img_channels)
            T_ag = torch.cat(ag_img_channels)
        
        T = torch.cat([T_ab, T_ag], dim=0)
        
        return T, tensor(label, dtype=torch.float32), id_


class EarlyStopping(object):
    """ Early stops the training if validation loss doesn't improve 
        after a given patience. 
    """
    
    def __init__(self, patience, min_change=0.0001, mode="min"):
        """
        Args:
            patience (int): How long to wait after last time 
            validation loss improved.
            min_delta (float): Minimum change in the monitored
            quantity to qualify as an improvement.
            mode (str): "max" or "min". If "max", a larger
            value is considered as improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_change = min_change
        self.mode = mode

    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        else:
            if self.mode == "min":
                if score > self.best_score - self.min_change:
                    self.counter += 1
                    print(
                        f"EarlyStopping counter:" \
                        f"{self.counter} out of {self.patience}"
                    )
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.counter = 0
            elif self.mode == "max":
                if score < self.best_score + self.min_change:
                    self.counter += 1
                    print(
                        f"EarlyStopping counter:" \
                        f"{self.counter} out of {self.patience}"
                    )
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.counter = 0


def kfold_cv(labels, *, cv=4, random_state=None, stratified=False):
    if stratified:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, 
                             random_state=random_state)
        for fold, (train_indices, val_indices) in \
                enumerate(kf.split(np.zeros(len(labels)), labels)):
            yield fold, (train_indices, val_indices)
    else:
        kf = KFold(n_splits=cv, shuffle=True, 
                   random_state=random_state)
        for fold, (train_indices, val_indices) in \
                enumerate(kf.split(range(len(labels)))):
            yield fold, (train_indices, val_indices)


def train_model(
        model: nn.Module,
        loader: DataLoader, 
        opt: torch.optim.Optimizer,
        device: torch.device,
        cutoff: float
):
    model.to(device).train()
    ce = nn.BCELoss()
    
    all_loss = []
    all_bac = []
    all_mcc = []
    
    pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, batch in pbar:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        scores = model(inputs).squeeze(-1)
        preds = [1 if s > cutoff else 0 for s in scores.tolist()]
        loss = ce(scores, labels)
        all_loss.append(loss.item())
        bac = balanced_accuracy_score(labels.cpu().numpy(), preds)
        all_bac.append(bac)
        mcc = matthews_corrcoef(labels.cpu().numpy(), preds)
        all_mcc.append(mcc)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        mean_loss = np.mean(all_loss)
        mean_bac = np.mean(all_bac)
        mean_mcc = np.mean(all_mcc)
        pbar.set_description_str(
            (f"Batch {i + 1} - loss: {mean_loss:.4f}, "
             f"bac: {mean_bac:.4f}, "
             f"mcc: {mean_mcc:.4f}")
        )

    return mean_loss, mean_bac, mean_mcc


@torch.no_grad()
def eval_model(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        cutoff: float
):
    model.to(device).eval()
    ce = nn.BCELoss()
    
    all_loss = []
    all_bac = []
    all_mcc = []
    all_scores = []
    for batch in loader:
        inputs = batch[0].to(device)  # [N,E]
        labels = batch[1].to(device)
        scores = model(inputs).squeeze(-1)
        all_scores.extend(scores.tolist())
        preds = [1 if s > cutoff else 0 for s in scores.tolist()]
        loss = ce(scores, labels)
        all_loss.append(loss.item())
        bac = balanced_accuracy_score(labels.cpu().numpy(), preds)
        all_bac.append(bac)
        mcc = matthews_corrcoef(labels.cpu().numpy(), preds)
        all_mcc.append(mcc)
    
    mean_loss = np.mean(all_loss)
    mean_bac = np.mean(all_bac)
    mean_mcc = np.mean(all_mcc)
    print(
        (f" - val_loss: {mean_loss:.4f}, "
         f"val_bac: {mean_bac:.4f}, "
         f"val_mcc: {mean_mcc:.4f}")
    )
    
    return all_scores, mean_loss, mean_bac, mean_mcc


def plot_metric(fname, train_metrics, val_metrics, metric):
    assert len(train_metrics) == len(val_metrics)
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, c="red")
    plt.plot(epochs, val_metrics, c="blue")
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["subtrain_" + metric, 'val_' + metric])
    plt.savefig(fname)
    plt.show()


def plot_roc_curve(fname, labels, preds_proba):
    fpr, tpr, thresholds = roc_curve(labels, preds_proba)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    roc_auc = roc_auc_score(labels, preds_proba)
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    ax = plt.gca()
    plt.axis("equal")
    plt.axis("square")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    disp.plot(ax=ax, c="green")
    ax.plot([0, 1], [0, 1], c="grey", linestyle="--")
    plt.savefig(fname)
    plt.close()
    return roc_auc, optimal_threshold


def plot_precision_recall_curve(fname, labels, preds_proba):
    precision, recall, _ = precision_recall_curve(labels, preds_proba)
    avg_precision = average_precision_score(labels, preds_proba)
    disp = PrecisionRecallDisplay(
        precision=precision, recall=recall, 
        average_precision=avg_precision
    )
    ax = plt.gca()
    plt.axis("equal")
    plt.axis("square")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    disp.plot(ax=ax, c="green")
    plt.savefig(fname)
    plt.close()
    return avg_precision


def record_training(fname, stopping_epoch, **metrics):
    tbl = pd.DataFrame()
    tbl["epoch"] = range(1, stopping_epoch+1)
    for metric, values in metrics.items():
        tbl[metric] = values
    tbl.to_csv(fname, index=False)


def write_eval_results(
        fname, 
        ids: Sequence[str], 
        targets: Sequence[int],
        scores: Sequence[float],
        flags
):
    tbl = pd.DataFrame()
    tbl["id"] = ids
    tbl["flags"] = flags
    tbl["score"] = scores
    tbl["pred"] = [int(s>0.5) for s in scores]
    tbl["target"] = targets
    tbl["result"] = tbl["pred"] == tbl["target"]
    tbl.to_csv(fname, index=False)
    return tbl


@torch.no_grad()
def get_result_on_existing_model(
        model: nn.Module, 
        data: Union[Dataset, Tensor],
        model_files,
        device: torch.device = "cuda:0",
        cutoff: float = 0.5,
        batch_size=32
):
    all_scores = np.zeros((len(data),))
    
    if isinstance(data, Dataset):
        loader = DataLoader(data, batch_size=batch_size)
        # print("len_loader", len(loader))
        for pth in model_files:
            model.load_state_dict(torch.load(pth, map_location=device))
            scores, _, _, _ = eval_model(model, loader, device, cutoff)
            scores = np.array(scores)
            all_scores += scores / len(model_files)
        ids = data.ids
        tgts = data.targets
        flags = [data.get_sample_type(id_, tgt) for id_, tgt in zip(ids, tgts)]
        return ids, tgts, all_scores, flags
    elif isinstance(data, Iterable):
        data = data.to(device)
        for pth in model_files:
            model.load_state_dict(torch.load(pth, map_location=device))
            scores = model(data).squeeze(-1).cpu().numpy()  # [B]
            all_scores += scores / len(model_files)
        return all_scores
    else:
        raise ValueError(f"Invalid type for data: {type(data)}")