import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# Função para treinar e avaliar modelos de regressão e salvar o melhor como pickle
def rank_and_save_best_model(train_x, train_y, test_x, test_y, model_filename='best_model.pkl'):
    # Definir os modelos de regressão para avaliação
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(random_state=42),
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor()
    }

    # Inicializar variáveis para armazenar as métricas dos modelos
    metrics = []

    # Treinar e avaliar cada modelo
    for name, model in models.items():
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)

        # Calcular métricas: R², MAE e MAPE
        r2 = r2_score(test_y, predictions)
        mae = mean_absolute_error(test_y, predictions)
        mape = mean_absolute_percentage_error(test_y, predictions)

        # Armazenar os resultados
        metrics.append({
            'Model': name,
            'R2': r2,
            'MAE': mae,
            'MAPE': mape
        })

    # Converter os resultados em um DataFrame para melhor visualização
    metrics_df = pd.DataFrame(metrics)

    # Exibir as métricas para cada modelo
    print(metrics_df)

    # Selecionar o melhor modelo com base no R²
    best_model_name = metrics_df.sort_values(by='R2', ascending=False).iloc[0]['Model']
    best_model = models[best_model_name]

    # Treinar novamente o melhor modelo com todos os dados de treino para salvar
    best_model.fit(train_x, train_y)

    # Salvar o melhor modelo como um arquivo pickle
    with open(f'models/{model_filename}', 'wb') as f:
        pickle.dump(best_model, f)

    print(f"O melhor modelo é: {best_model_name}, e foi salvo como {model_filename}")