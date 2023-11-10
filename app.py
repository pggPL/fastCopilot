import pickle

import torch
from flask import Flask, jsonify, request, render_template

from model import CharRNN, sample

app = Flask(__name__)
app.debug = True

def load_model(model_name):
    # load chars
    unpickled = pickle.load(open(f'models/{model_name}/chars.pkl', 'rb'))
    int2char = unpickled['int2char']
    char2int = unpickled['char2int']
    
    # compute list of chars
    chars = list(int2char.values())
    
    
    # load model weights
    model = CharRNN(chars, int2char, char2int)
    model.load_state_dict(torch.load(f'models/{model_name}/model.pth', map_location=torch.device('cpu')))
    
    return {
        'model': model,
        'chars': chars,
        'int2char': int2char,
        'char2int': char2int
    }
    

# Load models into one dictionary
models = {
    'English poetry': load_model('poems_small'),
    'Polish sentences': load_model('polish_small'),
    'Linux source': load_model('linux'),
}

@app.route('/')
def index():
    model_names = list(models.keys())
    return render_template('index.html', models=model_names)


@app.route('/predict', methods=['POST'])
def predict():
    # random 3 character
    import string
    
    src = request.get_json()['content']
    model = models[request.get_json()['model']]['model']
    
    
    # remove all non-alphabetical characters
    src = ''.join([c for c in src if c in model.chars])
    
    prediction = sample(model, len(src) + 20, prime=src)[len(src):]
    
    return jsonify({'prediction': prediction})

# about
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
