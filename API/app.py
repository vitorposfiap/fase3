from flask import Flask, jsonify
from tinydb import TinyDB

app = Flask(__name__)

# Conectar ao banco de dados TinyDB
db = TinyDB('../data/database/training_db.json')

# Endpoint para listar todos os usuários
@app.route('/data', methods=['GET'])
def get_usuarios():
    usuarios = db.all()
    return jsonify(usuarios)

if __name__ == '__main__':
    app.run(debug=True)