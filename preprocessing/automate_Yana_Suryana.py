import os
import zipfile
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler

def load_data():
    api = KaggleApi()
    api.authenticate()
    
    competition = 'house-prices-advanced-regression-techniques'
    download_path = '../house-prices-dataset_raw'
    
    if not os.path.exists(os.path.join(download_path, 'train.csv')):
        print("Data tidak ditemukan. Mengunduh dari Kaggle...")
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        
        api.competition_download_files(competition, path=download_path)
        
        zip_file = os.path.join(download_path, f'{competition}.zip')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print("Download selesai.")
    else:
            print("Data sudah tersedia, lanjut ke preprocessing.")
        
    return pd.read_csv(os.path.join(download_path, 'train.csv'))

def preprocess_data(df):
    cat_cols_with_na = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
    ]
    
    for col in cat_cols_with_na:
        df[col] = df[col].fillna('None')
        
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())
    
    df = df.drop_duplicates()
    
    df['SalePrice'] = np.log1p(df['SalePrice'])
    
    q1 = df['GrLivArea'].quantile(0.25)
    q3 = df['GrLivArea'].quantile(0.75)
    iqr = q3 - q1
    df = df[(df['GrLivArea'] >= (q1 - 1.5 * iqr)) & (df['GrLivArea'] <= (q3 + 1.5 * iqr))]
    
    bins = [1870, 1900, 1920, 1940, 1960, 1980, 2000, 2025]
    df['YearBuilt_Bin'] = pd.cut(df['YearBuilt'], bins=bins, labels=False)
    
    df = pd.get_dummies(df)
    
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice')
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df

if __name__ == "__main__":
    print("Memulai proses otomatis...")
    df_raw = load_data()
    df_clean = preprocess_data(df_raw)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'house-price-clean_preprocessing.csv')
    
    df_clean.to_csv(output_path, index=False)
    print(f"Preprocessing selesai! Data tersimpan di '{output_path}'. Bentuk data: {df_clean.shape}")