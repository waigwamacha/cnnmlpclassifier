import argparse, os, sys
import torch, gc

import numpy as np
import pandas as pd

from collections import Counter
from datetime import datetime
from pyprojroot import here
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm 

sys.path.insert(0, f"{here()}/src")
from cnn_classifier import CNNMLP
from dataset import MRIDataset
from utils import create_three_year_bins, training_loss_visualization

hostname = os.uname()[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
proj_dir = here()

#torch.manual_seed(42)
gc.collect()
torch.cuda.empty_cache()

if hostname == "worf":
    print(f"Hostname is: {hostname}, training on: {device}")
    traindir = "/shared/uher/Murage/BrainAgefMRI/data/train/"
    validationdir = "/shared/uher/Murage/BrainAgefMRI/data/validation/"
    phenocsv = "/shared/uher/Murage/BrainAgefMRI/data/mri_phenotype_partitioned.csv" 
else:
    print(f"Hostname is: {hostname}, training on: {device}")
    traindir = "/home/murage/Documents/data/train/" 
    validationdir = "/home/murage/Documents/data/validation/" 
    phenocsv = "/home/murage/Documents/data/mri_phenotype_partitioned.csv" 


def preptraindata(traindir=traindir, validationdir=validationdir, phenocsv=phenocsv):
#Load Training Data

    df = pd.read_csv(phenocsv)
    df['filename'] = df['filename'] + '.gz'
    df['filename'] = df['filename'].str.strip()

    df['scan_age_z'] = (df['scan_age'] - df['scan_age'].mean()) / df['scan_age'].std()
    print(f"Train mean: {df.scan_age.mean()}, std {df.scan_age.std()}, Standardized mean: {df['scan_age_z'].mean()}")
    
    df = create_three_year_bins(df)
    # Save train and validation filenames into list
    train = df[df['split'] == 'TRAINING']
    validation = df[df['split'] == 'VALIDATE']
    train_ids = train['filename'].to_list()
    validation_ids = validation['filename'].to_list()

    #print(f" Train N: {len(train_ids)} | Validation N: {len(validation_ids)}") 
    #!ls /home/murage/Documents/data/train/ | wc -l
    #!ls /home/murage/Documents/data/validation/ | wc -l

    train = train[['filename', 'scan_age', 'scan_age_z', 'sex', 'age_bracket_class', 'project']]
    validation = validation[['filename', 'scan_age', 'scan_age_z', 'sex', 'age_bracket_class', 'project']]

    #Balance classes in each batch
    class_counts = np.bincount(train['age_bracket_class'])
    class_weights = 1. / class_counts
    sample_weights = [class_weights[c] for c in train['age_bracket_class']]
    
    counts = Counter(train['age_bracket_class'])
    total = sum(counts.values())
    class_weights2 = [total / (10 * counts[i]) for i in range(10)]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = MRIDataset(df=train, id_col='filename', agebracket='age_bracket_class', target_col='scan_age_z', root_dir=traindir)
    validation_dataset = MRIDataset(df=validation, id_col='filename', agebracket='age_bracket_class', target_col='scan_age_z', root_dir=validationdir)

    train_dataloader = DataLoader(train_dataset, batch_size=4, pin_memory=True) #sampler=sampler,
    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

    return train_dataloader, validation_dataloader, torch.tensor(class_weights2, dtype=torch.float32)

train_dataloader, validation_dataloader, class_weights = preptraindata()


model = CNNMLP(input_shape=1, flattened_dim=18432).to(device)

classificationloss = nn.CrossEntropyLoss(weight = class_weights).to(device)
mae = nn.L1Loss().to(device)
mse = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)#lr=0.001)
#scheduler = LambdaLR(optimizer, lr_lambda)
trainset_agemean = torch.tensor(np.array([1.577888101618543e-16])) #15.427754
trainset_agemean = trainset_agemean.to(device)

class_epochs = 10

def trainmlpclassifier(model=model, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
                    optimizer=optimizer, device=device, num_epochs=30, name='cnnmlpclassifier'):
    
    datem = datetime.today().strftime("%Y-%m-%d")
    start = datetime.now()
    best_val_loss = float('inf')
    best_model_state = None
    tracking_loss, tracking_val_loss, tracking_mean_trainloss, tracking_mean_valloss = [], [], [], []

    patience = 7  # Number of epochs to wait for improvement
    trigger_count = 0
    beta, alpha = 0.3, 0.0
    lambda_reg = 0.01

    for epoch in tqdm(range(num_epochs)):

        print(f'Epoch {epoch+1}/{num_epochs}:\n -----------')

        model.train()
        train_loss, train_loss2, train_lossmean, val_lossmean = 0.0, 0.0, 0.0, 0.0
        train_accuracy = 0.0
        batch_n = 0
        for image, ageclass, labels in train_dataloader:
            batch_n += 1
            image, ageclass, labels = image.to(device), ageclass.to(device), labels.to(device)
            ageclass = ageclass.long()

            optimizer.zero_grad()
            predicted_age, predicted_logits = model(image)
            loss1 = mae(predicted_age, labels)
            baseloss = mae(trainset_agemean, labels)
            loss2 = classificationloss(predicted_logits, ageclass)

            # Apply L1 regularization:
            #l1_norm = sum(p.abs().sum() for p in model.parameters())
            #loss1 += lambda_reg * l1_norm
        
            # Apply L2 regularization: https://www.geeksforgeeks.org/machine-learning/l1l2-regularization-in-pytorch/
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss1 += lambda_reg * l2_norm

            if epoch < class_epochs:
                batch_loss = loss2 + 0.05 * loss1
            else:
                batch_loss = 0.3 * loss1 + 0.7 * loss2

            #combinedloss = beta*loss1 + alpha*loss2
            #combinedloss.backward()
            batch_loss.backward()
            optimizer.step()

            #predicted_class = torch.softmax(predicted_logits, dim=1).argmax(dim=1)
            #train_acc = accuracy_fn(y_true=ageclass,y_pred=predicted_class)

            train_loss += loss1.item()
            train_loss2 += loss2.item()
            train_lossmean += baseloss
            #train_accuracy += train_acc
            #tracking_loss[(epoch, batch_n)] = float(loss1)
        tracking_loss.append(train_loss /len(train_dataloader))
        tracking_mean_trainloss.append(train_lossmean /len(train_dataloader))

        avg_train_loss =train_loss /len(train_dataloader)
        #avg_train_loss2 =train_loss2 /len(train_dataloader)
        avg_train_accuracy =train_accuracy /len(train_dataloader)
        print(f"Train loss (acc): {avg_train_loss:.2f}") #({avg_train_accuracy:.2f}%) | l2: {avg_train_loss2:.2f}")

        # Validation
        model.eval()
        val_loss, val_loss2 = 0.0, 0.0
        val_accuracy = 0.0
        
        with torch.no_grad():
            for vimages, vclass, vlabels in validation_dataloader:
                vimages, vclass, vlabels = vimages.to(device), vclass.to(device), vlabels.to(device)
                vclass = vclass.long()

                vpredictionsage, vpredicted_logits = model(vimages)
                vloss = mae(vpredictionsage, vlabels)
                basevalloss = mae(trainset_agemean, vlabels)
                vloss2 = classificationloss(vpredicted_logits, vclass)
                
                #vrmse = torch.sqrt(mse(vpredictionsage, vlabels))
                #val_rmse += vrmse

                #vpredicted_class = torch.softmax(vpredicted_logits, dim=1).argmax(dim=1)
                #val_acc = accuracy_fn(y_true=vclass, y_pred=vpredicted_class)
                val_loss += vloss.item()
                val_lossmean += basevalloss
                #val_loss2 += vloss2.item()
                #val_accuracy += val_acc

            tracking_val_loss.append(val_loss /len(validation_dataloader))
            tracking_mean_valloss.append(val_lossmean /len(validation_dataloader))
            avg_val_loss = val_loss / len(validation_dataloader)
            #avg_val_loss2 = val_loss2 / len(validation_dataloader)
            #avg_val_accuracy = val_accuracy / len(validation_dataloader)

            print(f"Val Loss (acc): {avg_val_loss:.2f}") #({avg_val_accuracy:.2f}%) | l2: {avg_val_loss2:.2f} """) #| RMSE: {avg_val_rmse:.4f} 

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                print(f"Model saved with validation MAE: {avg_val_loss:.4f}") # , accuracy: {avg_val_accuracy:.4f}% ")

            """ else:
                trigger_count += 1
                if trigger_count >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break """

    # Load best model weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if hostname == 'worf':
            try: 
                torch.save(model.state_dict(), f=f"/shared/uher/Murage/cnnmlp/models/{datem}_best_{name}_model.pth")
            except:
                print("Path: /shared/uher/Murage/cnnmlp/models/ not found")
        else:
            try: 
                torch.save(model.state_dict(), f=f"/home/murage/Documents/repos/cnnmlp/models/{datem}_best_{name}_model.pth")
            except:
                print("Path: /home/murage/Documents/repos/ not found")

    tracking_loss = [torch.tensor(t).cpu() for t in tracking_loss]
    tracking_val_loss = [torch.tensor(t).cpu() for t in tracking_val_loss]
    tracking_mean_trainloss = [torch.tensor(t).cpu() for t in tracking_mean_trainloss]
    tracking_mean_valloss = [torch.tensor(t).cpu() for t in tracking_mean_valloss] 
    
    training_loss_visualization(tracking_loss=tracking_loss, trackingval_loss=tracking_val_loss, 
                            tracking_mean_trainloss=tracking_mean_trainloss, tracking_mean_valloss=tracking_mean_valloss)
    end = datetime.now()
    
    print(f"Took {end-start} to run")

    return tracking_loss, tracking_val_loss, tracking_mean_trainloss, tracking_mean_valloss


def main():

    parser = argparse.ArgumentParser(description="Load Data and Train CNN-MLP-Classifier")
    parser.add_argument("--Train Directory", required=False, 
                        help="Train data directory")
    parser.add_argument("--Validation Directory", required=False, help="Validation data directory")
    parser.add_argument("--Phenotype File", required=False, help="CSV file with phenotypes for train data")
    args = parser.parse_args()

    preptraindata()
    trainmlpclassifier()

    parser.exit(status=0, message="Right on: Data loaded succesfully, training mlpclassifier ... \n")

if __name__ == '__main__':

    main()