
import os, glob, sys, argparse
import torch, gc

from pyprojroot import here
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, f"{here()}/src")
from cnn_classifier import CNNMLP
from dataset import MRIDataset
from data_setup import generate_phenofile
from utils import accuracy_fn

gc.collect()
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

hostname = os.uname()[1]
if hostname == "worf":
    testdatadir = "/shared/uher/Murage/BrainAgefMRI/data/test/"
    #phenocsv = "/shared/uher/Murage/BrainAgefMRI/data/test/mri_phenotype_partitioned.csv" 
else:
    testdatadir = "/home/murage/Documents/data/test/" 
    #phenocsv = "/home/murage/Documents/data/mri_phenotype_partitioned.csv" 

mae = nn.L1Loss()
mse = nn.MSELoss()
classificationloss = nn.CrossEntropyLoss()
loaded_model = CNNMLP(input_shape=1, flattened_dim=18432).to(device)

def create_testdataloader(dataset:str, batch_n=8):
    test_data_dir=f"{testdatadir}/{dataset.upper()}/"
    df = generate_phenofile(dataset)
    print(df.shape)
    #print(df.columns)
    #print(f"{df.age_bracket_class.describe()}")
    test_dataset = MRIDataset(df=df, id_col='filename', agebracket='age_bracket_class', target_col='scan_age_z', root_dir=test_data_dir)
    for i in range(3):
        try:
            image, ageclass, label = test_dataset[i]
            print(f"Index: {i}, {ageclass}, Image shape: {image.shape if hasattr(image, 'shape') else type(image)}, Label: {label}")
        except KeyError:
            print(f"Index: {i}, {ageclass}, Image shape: {image.shape if hasattr(image, 'shape') else type(image)}, Label: {label}")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_n, shuffle=False, pin_memory=True)
    return df, test_dataloader

trained_models = glob.glob(f"{here()}/models/*.pth")
print(f"Found {len(trained_models)} models, using {trained_models[0]}")

for model in trained_models:
    print(f"Model: {model}")

loaded_model.load_state_dict(torch.load(trained_models[0], weights_only=True))

def predictbrainage(test_dataloader: torch.utils.data.DataLoader, 
            loss_fn= mae, model=loaded_model, accuracy_fn=accuracy_fn, 
            device: torch.device = device):
    
    loss, rmse, overall_accuracy = 0, 0, 0
    scan_age, brain_age, test_accuracy = [], [], []
    model.eval()

    with torch.inference_mode():
        for image, ageclass, label in test_dataloader:
            image, ageclass, label = image.to(device), ageclass.to(device), label.to(device)
            ageclass = ageclass.long()

            predicted_age, predicted_logits = model(image)
            scan_age.append(label.cpu())
            brain_age.append(predicted_age.cpu())

            loss += mae(predicted_age, label)
            loss2 = classificationloss(predicted_logits, ageclass)
            rmse += torch.sqrt(mse(label, predicted_age))

            tpredicted_class = torch.softmax(predicted_logits, dim=1).argmax(dim=1)
            test_acc = accuracy_fn(y_true=ageclass,
                        y_pred=tpredicted_class)
            #test_accuracy.append(test_acc.cpu().item())
                
            
        # Scale loss and acc
        loss /= len(test_dataloader)
        rmse /= len(test_dataloader)
        #overall_accuracy /= len(test_dataloader)

    scan_age = torch.cat(scan_age).numpy()
    brain_age = torch.cat(brain_age).numpy()
    #test_accuracy = torch.cat(test_accuracy).numpy()
    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_rmse": rmse.item(),
            "age": scan_age,
            "predicted": brain_age,
            #"accuracy (%)": test_accuracy
            }


def main():

    parser = argparse.ArgumentParser(description="Predict BrainAge on Test Data using CNN-MLP-Classifier")
    parser.add_argument("--Test Dataloader", required=True, 
                        help="Test dataloader")
    parser.add_argument("--Loss_fn", required=False)
    parser.add_argument("--Model", required=False)
    args = parser.parse_args()

    predictbrainage()

    parser.exit(status=0, message="Right on: Data laoded succesfully, training mlpclassifier ... \n")

if __name__ == '__main__':

    main()