import torch
import torch.nn as nn
import numpy as np

int2char = dict(enumerate('abcdefghijklmnopqrstuvwxyz '))
char2int = {ch: ii for ii, ch in int2char.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super(CharRNN, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # Creating character dictionaries
        self.chars = tokens
        self.int2char = int2char
        self.char2int = char2int

        # Define the layers of the model
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                          batch_first=True, dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden


def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)  # corrected line

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

def predict(model, char, hidden=None, temperature=1.0):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state. '''
    # One-hot encode the input character
    x = np.array([[model.char2int[char]]])
    x = one_hot_encode(x, len(model.chars))
    x = torch.from_numpy(x).to(device)

    # Detach hidden state from history
    hidden = (hidden[0].data, hidden[1].data)
    # Get the output of the model
    out, hidden = model(x, hidden)

    # Apply temperature to softmax function
    out_dist = out.data.view(-1).div(temperature).exp()
    top_i = torch.multinomial(out_dist, 1)[0]

    # Return the encoded value of the predicted char and the hidden state
    predicted_char = model.int2char[top_i.item()]

    return predicted_char, hidden

def sample(model, size, prime=' ', temperature=1.0):
    # Model in evaluation mode
    model.eval()

    # Run through the prime characters
    chars = [ch for ch in prime]
    hidden = (model.init_hidden(1)[0].to(device), model.init_hidden(1)[1].to(device))
    for ch in prime:
        char, hidden = predict(model, ch, hidden, temperature=temperature)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for _ in range(size - len(prime)):
        char, hidden = predict(model, chars[-1], hidden, temperature=temperature)
        chars.append(char)

    return ''.join(chars).replace('\n', '\n')