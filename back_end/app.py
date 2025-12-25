# coding=utf-8
from flask import Flask, request, jsonify
from flask_cors import CORS

from handler import query_handler
from two_stage import two_stage_qa

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return 'server running'


@app.route('/query_v2', methods=['POST'])
def query_v2():
    question = request.get_json()["question"]
    result = two_stage_qa(question)
    return jsonify(result)


@app.route('/query', methods=['POST'])
def query():
    question = request.get_json()["question"]
    return query_handler(question)


if __name__ == "__main__":
    app.run(port=5001, debug=True)