from flask import Flask, jsonify, request, render_template

from model import CharRNN, sample

app = Flask(__name__)
app.debug = True


@app.route('/')
def index():
    print("aaa")
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # random 3 character
    import random
    import string
    
    src = request.get_json()['content']
    
    chars = string.ascii_lowercase + ' '
    model = CharRNN(chars, 10, 3)
    
    # remove all non-alphabetical characters
    src = ''.join([c for c in src if c in chars])
    
    prediction = sample(model, len(src) + 20, prime=src)[len(src):]
    
    return jsonify({'prediction': prediction})

# about
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
