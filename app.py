from flask import Flask, render_template, request
from engine import *

app = Flask(__name__)
# edits = Edits(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def query_form():
    text = (request.form['text'],request.form['text2'])
    sentences_list = text[0].split('.')
    topsimilar = get_top_similar(text[1], sentences_list, similarity_matrix, 5)
    return render_template('results.html',value=topsimilar)

if __name__ == '__main__':
    app.run()

