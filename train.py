from typing import *

import click
import torch
from imapep.ml import *
from sklearn.metrics import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset


@click.command()
@click.option("--train-dir", required=True)
@click.option("--model-dir", required=True)
@click.option("--shape", default=100)
@click.option("--channel-mode", default="OOO")
@click.option("--device", default="cuda:1")
@click.option("--num-folds", default=10)
@click.option("--max-epochs", default=100)
@click.option("--lr", default=8e-4)
@click.option("--batch-size", default=32)
@click.option("--cutoff", default=0.5)
@click.option("--stop-patience", default=5)
def train(train_dir, model_dir, shape, channel_mode, device, num_folds, max_epochs, lr, batch_size, cutoff, stop_patience):
    side_len = shape // 4
    model_args = {"conv_in": 6, "conv_out": 32, "fc_dim": 32*side_len**2*2, "dropout": 0.75, "bias": True}
    earlystop_args = {"patience": stop_patience, "min_change": 0.002, "mode": "min"}
    train_set = ProteinDataset(train_dir, device, shape, channel_mode)
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)

    # K-fold CV
    for fold, (subtrain_indices, val_indices) in kfold_cv(train_set.targets, cv=num_folds, random_state=42, stratified=True):
        fold += 1
        model = CNN(**model_args).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, verbose=True)
        early_stopping = EarlyStopping(**earlystop_args)
        # Prepare datasets
        subtrain_set = Subset(train_set, subtrain_indices)
        val_set = Subset(train_set, val_indices)
        subtrain_loader = DataLoader(subtrain_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        all_subtrain_loss = []
        all_val_loss = []
        all_subtrain_bac = []
        all_val_bac = []
        all_subtrain_mcc = []
        all_val_mcc = []
        epoch = 1  # indispensible

        for epoch in range(1, max_epochs+1):
            print(f"Fold{fold}, Epoch{epoch}")
            # Train for an epoch
            subtrain_loss, subtrain_bac, subtrain_mcc = train_model(model, subtrain_loader, opt, device, cutoff)
            _, val_loss, val_bac, val_mcc = eval_model(model, val_loader, device, cutoff)
            # Collect result data
            all_subtrain_loss.append(subtrain_loss)
            all_val_loss.append(val_loss)
            all_subtrain_bac.append(subtrain_bac)
            all_val_bac.append(val_bac)
            all_subtrain_mcc.append(subtrain_mcc)
            all_val_mcc.append(val_mcc)
            # Schedule and early-stop if necessary
            scheduler.step(subtrain_loss)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"[Early-stopping epochs]: {epoch}")
                break
        
        torch.save(model.state_dict(), f"{model_dir}/model{fold}.pth")


if __name__ == "__main__":
    train()