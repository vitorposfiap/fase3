import pandas as pd
import requests

def api_call(hostname):
    # URL do endpoint da API
    url = hostname

    # Fazer a requisição GET para o endpoint
    response = requests.get(url)

    # Verificar se a requisição foi bem-sucedida
    if response.status_code == 200:

        # Converter a resposta JSON para um DataFrame do pandas
        data = response.json()
        df = pd.DataFrame(data)

        return df