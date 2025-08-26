
import argparse, glob, os, sys 
import pandas as pd

from pyprojroot import here
sys.path.insert(0, f"{here()}/src")
from utils import create_three_year_bins


hostname = os.uname()[1]
if hostname == "worf":
    print(f"Hostname is: {hostname}")
    datadir = "/shared/uher/Murage/BrainAgefMRI/data/test/"
    #phenocsv = "/shared/uher/Murage/BrainAgefMRI/data/test/mri_phenotype_partitioned.csv" 
else:
    print(f"Hostname is: {hostname}")
    datadir = "/home/murage/Documents/data/test/" 
    #phenocsv = "/home/murage/Documents/data/mri_phenotype_partitioned.csv" 


def generate_phenofile(dataset:str, dir=datadir):

    dataset = dataset.upper()

    csv_files = glob.glob(f"{dir}/{dataset}/*.csv")
    print(f"Found {len(csv_files)} csv files")
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV file found in the directory.")
    phenofile = csv_files[0]  # Use the first found CSV file
    df = pd.read_csv(phenofile)
    #print(df.shape)
    #print(df.columns)

    processed_subjects = []

    for file in os.listdir(f"{dir}/{dataset}/"): 
        if file.endswith('.gz'):
            processed_subjects.append(file)

    df_images = pd.DataFrame(processed_subjects, columns=['participant'])

    if dataset == 'FORBOW':
        df_images[['participant_id', 'meh', 'meh2']] = df_images['participant'].str.split('.', expand=True)
        dfT1 = df_images[['participant_id']]
        print(f"T1 shape: {dfT1.shape}, Pheno shape: {df.shape}")

        dfT1 = dfT1.assign(filename=dfT1['participant_id'] + '.nii.gz')
        
        df['participant_id'] = df['subject_id'].str.replace('_', '')
        df = df[['participant_id',  'sex', 'scan_age']]
        dfT1.loc[: ,'participant_id'] = dfT1.loc[:, 'participant_id'].str.replace('FORBOW-', '')

        #Merge Image files with pheno data
        df.columns = df.columns.str.strip()
        dfT1.columns = dfT1.columns.str.strip()
        df = df.merge(dfT1, on='participant_id', how='left')

        missing_frb = df[df['filename'].isna()]
        print(f"df shape: {df.shape}, Missing n shape: {missing_frb.shape}")

        frb_df = df.merge(dfT1, on='participant_id', how='inner')
        
        df = create_three_year_bins(frb_df)
        df.rename(columns={'filename_x': 'filename'}, inplace=True)
        #create standardized age values
        df['scan_age_z'] = (df['scan_age'] - 15.427753927254452) / 5.778836817291233
        print(f"Final df shape: {df.shape}")

    elif dataset == 'ABCD':

        df['filename'] = 'ABCD-sub-' + df['participant_id'] + '.nii.gz'
        abcd_ = pd.DataFrame(processed_subjects, columns=['filename'])
        abcd = df.merge(abcd_, on='filename', how='inner')
        df = create_three_year_bins(abcd)
        df['scan_age_z'] = (df['scan_age'] - 15.427753927254452) / 5.778836817291233

    elif dataset == 'BHRC':

        df_images[['participant_id', 'meh', 'meh2']] = df_images['participant'].str.split('.', expand=True)
        bhrcT1 = df_images[['participant_id']]
        bhrcT1 =bhrcT1.assign(filename=bhrcT1['participant_id'] + '.nii.gz')
        bhrcT1['scan_id'] = bhrcT1['participant_id'].str.split('-').str[-1]
        bhrcT1['scan_id'] = bhrcT1['scan_id'].astype(int)
        df = df.merge(bhrcT1, on='scan_id', how='left')
        df.rename(columns={'filename_x': 'filename'}, inplace=True)
        df = df[df['filename'].notna()]
        df = create_three_year_bins(df)
        df['scan_age_z'] = (df['scan_age'] - 15.427753927254452) / 5.778836817291233

    elif dataset == 'BHRC2':

        df_images[['participant_id', 'meh', 'meh2']] = df_images['participant'].str.split('.', expand=True)
        bhrcT1 = df_images[['participant_id']]
        bhrcT1 =bhrcT1.assign(filename=bhrcT1['participant_id'] + '.nii.gz')
        bhrcT1['scan_id'] = bhrcT1['participant_id'].str.split('-').str[-1]
        bhrcT1['scan_id'] = bhrcT1['scan_id'].astype(int)
        df = df.merge(bhrcT1, on='scan_id', how='left')
        df.rename(columns={'filename_x': 'filename'}, inplace=True)
        df = df[df['filename'].notna()]
        df = create_three_year_bins(df)
        df['scan_age_z'] = (df['scan_age'] - 15.427753927254452) / 5.778836817291233

    elif dataset == 'BHRC3':

        df_images[['participant_id', 'meh', 'meh2']] = df_images['participant'].str.split('.', expand=True)
        bhrcT1 = df_images[['participant_id']]
        bhrcT1 =bhrcT1.assign(filename=bhrcT1['participant_id'] + '.nii.gz')
        bhrcT1['scan_id'] = bhrcT1['participant_id'].str.split('-').str[-1]
        bhrcT1['scan_id'] = bhrcT1['scan_id'].astype(int)
        df = df.merge(bhrcT1, on='scan_id', how='left')
        df.rename(columns={'filename_x': 'filename'}, inplace=True)
        df = df[df['filename'].notna()]
        df = create_three_year_bins(df)
        df['scan_age_z'] = (df['scan_age'] - 15.427753927254452) / 5.778836817291233


    return df


def main():

    parser = argparse.ArgumentParser(description="Load Data and Train CNN-MLP-Classifier")
    parser.add_argument("--Dataset", required=True, help="name of test dataset (e.g., bhrc)")
    parser.add_argument("--Dir", required=False, help="Top level test data directory")
    args = parser.parse_args()

    generate_phenofile()

    parser.exit(status=0, message="Right on: Data loaded succesfully, training mlpclassifier ... \n")

if __name__ == '__main__':

    main()