import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Função para remover outliers usando Isolation Forest e salvar o modelo
def remove_outliers_isolation_forest(df, contamination=0.05, model_filename='models/isolation_forest_model.pkl'):
    # Inicializar o modelo IsolationForest
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    
    # Ajustar o modelo aos dados e prever se os pontos são outliers
    outliers = isolation_forest.fit_predict(df.select_dtypes(include=[np.number]))
    
    # Salvar o modelo IsolationForest em um arquivo pickle
    with open(model_filename, 'wb') as f:
        pickle.dump(isolation_forest, f)
    
    # Manter apenas os pontos que não são outliers
    df_clean = df[outliers == 1].reset_index(drop=True)
    
    return df_clean

# Função para dividir o dataset em treino e teste, aplicar o StandardScaler e salvar os pesos
def prepare_train_test(df, target_column='Exam_Score', test_size=0.2, random_state=42, scaler_filename='models/scaler_weights.pkl'):
    # Dividir os dados em features (X) e target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Dividir os dados em treino e teste
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Inicializar o StandardScaler
    scaler = StandardScaler()
    
    # Ajustar o scaler nos dados de treino e transformar os dados de treino
    train_x_scaled = scaler.fit_transform(train_x)
    
    # Salvar os pesos do scaler em um arquivo pickle
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Aplicar os mesmos pesos nos dados de teste
    test_x_scaled = scaler.transform(test_x)
    
    return train_x_scaled, train_y, test_x_scaled, test_y