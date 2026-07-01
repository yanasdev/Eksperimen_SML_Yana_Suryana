import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.preprocessing import StandardScaler

def load_data():
    """Mengunduh dan memuat dataset."""
    path = kagglehub.competition_download('house-prices-advanced-regression-techniques')
    file_path = os.path.join(path, 'train.csv')
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Fungsi utama untuk melakukan preprocessing sesuai tahapan."""
    
    # 1. Menangani Data Kosong
    cat_cols_with_na = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in cat_cols_with_na:
        df[col] = df[col].fillna('None')
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    
    # 2. Menghapus Data Duplikat
    df = df.drop_duplicates()
    
    # 3. Normalisasi atau Standarisasi Fitur
    df['SalePrice'] = np.log1p(df['SalePrice'])
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice')
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # 4. Deteksi dan Penanganan Outlier
    Q1 = df['GrLivArea'].quantile(0.25)
    Q3 = df['GrLivArea'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['GrLivArea'] >= (Q1 - 1.5 * IQR)) & (df['GrLivArea'] <= (Q3 + 1.5 * IQR))]
    
    # 5. Encoding Data Kategorikal
    df = pd.get_dummies(df)
    
    # 6. Binning
    bins = [1870, 1900, 1920, 1940, 1960, 1980, 2000, 2025]
    df['YearBuilt_Bin'] = pd.cut(df['YearBuilt'], bins=bins, labels=False)
    
    return df

if __name__ == "__main__":
    print("Memulai proses otomatis...")
    df_raw = load_data()
    df_clean = preprocess_data(df_raw)
    
    # Simpan hasil ke CSV agar bisa digunakan di tahap modelling
    df_clean.to_csv('clean_data.csv', index=False)
    print(f"Preprocessing selesai! Data tersimpan di 'clean_data.csv'. Bentuk data: {df_clean.shape}")