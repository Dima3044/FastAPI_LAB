import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def preprocess(df):
    df['shell_ratio'] = df['shell_weight'] / (df['whole_weight'] + 1e-6)
    df['volume'] = df['length'] * df['diameter'] * df['height']
    
    weight_cols = ['whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']
    for col in weight_cols + ['volume']:
        df[col] = np.sqrt(df[col])
        
    df['sex_I'] = (df['sex'] == 'I').astype(int)
    df['sex_M'] = (df['sex'] == 'M').astype(int)
    df.drop('sex', axis=1, inplace=True)
    
    cols = [
        'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
        'viscera_weight', 'shell_weight', 'shell_ratio', 'volume', 'sex_I', 'sex_M'
    ]
    return df[cols]

def train():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    cols = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"]
    df = pd.read_csv(url, names=cols)
    
    # Очистка выбросов как в ноутбуке
    df = df[(df['height'] <= 0.250) & (df['rings'] <= 22)]
    
    y = df['rings']
    X = df.drop(columns=['rings'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_p = preprocess(X_train)
    X_test_p = preprocess(X_test)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_p)
    X_test_s = scaler.transform(X_test_p)
    
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    print(f"Train R2: {model.score(X_train_s, y_train):.4f}")
    print(f"Test R2:  {model.score(X_test_s, y_test):.4f}")
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✅ model.pkl и scaler.pkl сохранены")

if __name__ == "__main__":
    train()