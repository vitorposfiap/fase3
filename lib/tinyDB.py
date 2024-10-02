import csv
from tinydb import TinyDB

# Nome do arquivo com os dados de treinamento
csv_filename = '../data/raw/StudentPerformanceFactors.csv'

db = TinyDB('../data/database/training_db.json')

# Limpar o banco de dados antes de inserir novos registros
db.truncate()

# Ler os dados do arquivo CSV e inserir no banco de dados
with open(csv_filename, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)  # DictReader usa a primeira linha como nome das colunas
    for row in reader:
        # Inserir cada linha do CSV como um registro no banco de dados
        db.insert(dict(row))