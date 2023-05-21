import numpy as np 
import pandas as pd 
import random
import wandb

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import time
import math
import argparse
from types import SimpleNamespace
from copy import deepcopy

#--------------------------------------------------------

# Global constant variables used throughout the program
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device = ", device)

SOS_token = "@"
EOS_token = "#"
PAD_token = "^"
UNK_token = "$"

SOS_idx = 0
EOS_idx = 1
PAD_idx = 2
UNK_idx = 3

batch_size = 32
# languages in antarakshar_dataset
languages = ["asm", "ben", "brx", "guj", "hin", "kan", "kas", "kok", "mai", "mal", "mar", "mni", "ori", "pan", "san", "sid", "tam", "tel", "urd"]
# paths to save best model and predictions
best_model_path = '/script_model/best_model_script.pth'
test_pred_path = '/script_predictions/pred_script.csv'
#-------------------------------------------------------

# times the execution of training
def timeInMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    s = format(s, ".0f")
    return str(m) + "m " + str(s) + "s"

# class to prepare and store vocabulary
class Script:
    def __init__(self, name):
        self.name = name
        self.char2index = {SOS_token: SOS_idx, EOS_token: EOS_idx, PAD_token: PAD_idx, UNK_token: UNK_idx}
        self.char2count = {}
        self.index2char = {SOS_idx: SOS_token, EOS_idx: EOS_token, PAD_idx: PAD_token, UNK_idx: UNK_token}
        self.n_chars = 4  # Count SOS, EOS, PAD and UNK

    # function that adds characters in the word to the vocabulary
    def addWord(self, word):
        for char in word:
            self.addChar(char)
    # function that adds characters to the vocabulary
    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

# prepare vocabulary from given data having source and target words
def prepareVocab(data, in_scr="lat", out_scr="dev"):
    input_vocab = Script(in_scr)
    output_vocab = Script(out_scr)
    
    for pair in data:
        input_vocab.addWord(pair[0])
        output_vocab.addWord(pair[1])
    
    return input_vocab, output_vocab

# returns tensor from word after mapping each character in the word to an index according to the vocabulary
# appends sos or eos accordingly if required
def tensorFromWord(word, vocab, sos=False, eos=False):
    char_list = []
    if sos:
        char_list.append(vocab.char2index[SOS_token])
    for char in word:
        if char in vocab.char2index:
            char_list.append(vocab.char2index[char])
        else:
            char_list.append(vocab.char2index[UNK_token])
    if eos:
        char_list.append(vocab.char2index[EOS_token])
    char_tensor = torch.tensor(char_list, dtype=torch.long)
    return char_tensor

# function that takes a list of tensors of word representations and pads them
def processData(data, vocab, sos=False, eos=False):
    tensor_list = []
    for word in data:
        word_tensor = tensorFromWord(word, vocab, sos, eos)
        tensor_list.append(word_tensor)
    word_tensor_pad = pad_sequence(tensor_list, padding_value=PAD_idx, batch_first=True)
    return word_tensor_pad

# returns word from tensor by mapping index to character in vocabulary
def wordFromTensor(word_tensor, vocab):
    word = ""
    for idx in word_tensor:
        if idx == EOS_idx:
            break
        if idx >= UNK_idx:
            word += vocab.index2char[idx.item()]
    return word

# class Encoder that receives source sequence
class Encoder(nn.Module):
    def __init__(self, cell_type, input_size, embedding_size, hidden_size, num_layers, dp, bidir=False):
        super(Encoder, self).__init__()
        self.cell_type = cell_type
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dp)
        self.bidir = bidir
        
        # input embedding
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        # no dropout if number of layers in the cell is 1
        if self.num_layers == 1:
            dp = 0.0
        if self.cell_type == "RNN":
            self.cell = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout=dp, bidirectional=self.bidir)
        elif self.cell_type == "GRU":
            self.cell = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout=dp, bidirectional=self.bidir)
        elif self.cell_type == "LSTM":
            self.cell = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=dp, bidirectional=self.bidir)

    def forward(self, x):
        # x shape: (seq_len, b_sz)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_len, b_sz, embedding_size)
        
        # handle cases of LSTM and Bidirectional accordingly
        cell = None
        if self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.cell(embedding)
            # cell shape: (2*n_layers, b_sz, seq_len) in case of bidirectional=True
            # convert it to (1, b_sz, seq_len)
            if self.bidir:
                b_sz = cell.size(1)
                cell = cell.view(self.num_layers, 2, b_sz, -1)
                cell = cell[-1]
                cell = cell.mean(axis=0)
            else:
                # take last layer's cell value
                cell = cell[-1,:,:]
            cell = cell.unsqueeze(0)
        else:
            outputs, hidden = self.cell(embedding)
        # similarly for hidden as cell
        if self.bidir:
            b_sz = hidden.size(1)
            hidden = hidden.view(self.num_layers, 2, b_sz, -1)
            hidden = hidden[-1]
            hidden = hidden.mean(axis=0)
        else:
            hidden = hidden[-1,:,:]
        hidden = hidden.unsqueeze(0)
        # outputs shape: (seq_len, b_sz, hidden_size)

        return hidden, cell

# class decoder that taken hidden output form encoder and target sequence to produce predicted sequence
class Decoder(nn.Module):
    def __init__(
        self, cell_type, input_size, embedding_size, hidden_size, output_size, num_layers, dp
    ):
        super(Decoder, self).__init__()
        self.cell_type = cell_type
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dp)

        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        if self.num_layers == 1:
            dp = 0.0
        if self.cell_type == "RNN":
            self.cell = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout=dp)
        elif self.cell_type == "GRU":
            self.cell = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout=dp)
        elif self.cell_type == "LSTM":
            self.cell = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=dp)
        # last fully connected layer that will produce output of size (1, b_sz, tgt_vocab_sz)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden, cell):
        # x shape: b_sz we want it to be (1, b_sz), one character at a ime
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, b_sz, embedding_size)

        if self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.cell(embedding, (hidden, cell))
        else:
            outputs, hidden = self.cell(embedding, hidden)
        # outputs shape: (1, b_sz, hidden_size)

        predictions = self.fc(outputs)

        # predictions shape: (1, b_sz, tgt_vocab_sz) to send it to
        # loss function we want it to be (b_sz, tgt_vocab_sz)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

# seq2seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # forward prop function that generates output of size(tgt_len, b_sz, tgt_vocab_sz)
    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_sz = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_size
        # tensor to store output of each step in sequence
        outputs = torch.zeros(target_len, batch_sz, target_vocab_size).to(device)

        # hidden and cell from encoder of size: (1, b_sz, seq_len)
        # repeat it fot number of decoder layers as the same output from the
        # last layer of encoder will be sent to each layer of decoder
        hidden, cell = self.encoder(source)
        hidden = hidden.repeat(self.decoder.num_layers,1,1)
        if self.decoder.cell_type == "LSTM":
            cell = cell.repeat(self.decoder.num_layers,1,1)

        # first input to the Decooder, <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store output
            outputs[t] = output

            # Take the argmax over the tgt_vocab_sz
            best_guess = output.argmax(dim=1)

            # Use Teacher forcing for half of the times (0.5) during training
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

# return number of equal predictions from a batch
def sum_accuracy(preds, target):
    num_equal_columns = torch.logical_or(preds == target, target == PAD_idx).all(dim=0).sum().item()
    return num_equal_columns

# runs the model in eval mode and returns accuracy and loss
def evaluateModel(model, dataloader, criterion, b_sz=32):
    model.eval()
    
    n_data = len(dataloader) * b_sz
    loss_epoch = 0
    n_correct = 0
    
    with torch.no_grad():
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq = input_seq.T.to(device)
            target_seq = target_seq.T.to(device)

            # Forward prop
            output = model(input_seq, target_seq, teacher_force_ratio=0.0)
            
            pred_seq = output.argmax(dim=2)
            n_correct += sum_accuracy(pred_seq, target_seq)

            # output is of shape: (tgt_len, b_sz, tgt_voab_sz)
            # convert it to (b_sz*tgt_len, tgt_vocab_sz) for calculating cross-entropy loss
            # similarly for target_seq
            output = output[1:].reshape(-1, output.shape[2])
            target = target_seq[1:].reshape(-1)
            
            loss = criterion(output, target)
            # sum of loss for each batch
            loss_epoch += loss.item()
        # report accuracy and loss for complete dataset
        acc = n_correct / n_data
        acc = acc * 100.0
        loss_epoch /= len(dataloader)
        return loss_epoch, acc

# same as evaluate but also saves the predictions into csv
def saveAndEvaluate(model, dataloader, criterion, df, vocab, b_sz=32):
    results = []
    model.eval()
    
    n_data = len(dataloader) * b_sz
    loss_epoch = 0
    n_correct = 0
    
    with torch.no_grad():
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            
            input_seq = input_seq.T.to(device)
            target_seq = target_seq.T.to(device)

            
            output = model(input_seq, target_seq, teacher_force_ratio=0.0)
            
            pred_seq = output.argmax(dim=2)
            n_correct += sum_accuracy(pred_seq, target_seq)

            output = output[1:].reshape(-1, output.shape[2])
            target = target_seq[1:].reshape(-1)
            
            loss = criterion(output, target)

            loss_epoch += loss.item()
            
            # find the predicted word from the pred_seq tensor
            pred_seq = pred_seq.T
            for idx in range(b_sz):
                word = wordFromTensor(pred_seq[idx], vocab)
                results.append(word)
                
        acc = n_correct / n_data
        acc = acc * 100.0
        loss_epoch /= len(dataloader)
        
        # save the predictions in a csv file
        df[2] = results
        new_column_names = {0: 'Source', 1: 'Target', 2: 'Predicted'}
        df = df.rename(columns=new_column_names)
        df.to_csv(test_pred_path, index=False)
        
        return loss_epoch, acc

# function that trains the model
def trainModel(model, criterion, optimizer, train_dataloader, valid_dataloader, num_epochs, batch_size=32):
    start = time.time()
    min_val_loss = 10000.0
    min_val_epoch = 0
    trigger = 0
    
    # stores the best model obtained in case of Early Stopping
    best_model_state = deepcopy(model.state_dict())
    
    tr_loss_list = []
    tr_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1} / {num_epochs}]")
        model.train()

        for batch_idx, (input_seq, target_seq) in enumerate(train_dataloader):
            
            input_seq = input_seq.T.to(device)
            target_seq = target_seq.T.to(device)

            
            output = model(input_seq, target_seq)
            
            output = output[1:].reshape(-1, output.shape[2])
            target = target_seq[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            # Back prop
            loss.backward()

            # Ensure that the gradients don't explode
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

        #-----------------------------------------------
        # Train loss and accuracy
        tr_loss, tr_acc = evaluateModel(model, train_dataloader, criterion, batch_size)
        tr_loss_list.append(tr_loss)
        tr_acc_list.append(tr_acc)
        
        print(f"Training Loss: {tr_loss:.2f}")
        print(f"Training Accuracy: {tr_acc:.2f}")

        #-----------------------------------------------
        # Valid loss and accuracy
        val_loss, val_acc = evaluateModel(model, valid_dataloader, criterion, batch_size)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        
        print(f"Validation Loss: {val_loss:.2f}")
        print(f"Validation Accuracy: {val_acc:.2f}")

        # Check for no decrease in validation loss for patience of 5 epochs
        if val_loss <= min_val_loss:
            best_model_state = deepcopy(model.state_dict())
            trigger = 0
            min_val_loss = val_loss
            min_val_epoch = epoch
        else:
            trigger += 1
        
        end = time.time()
        print("Time: ", timeInMinutes(end-start))
        print("----------------------------------")
        
        # if patience is over, then do early stopping
        if trigger == 5:
            print('Early stopping!')
            break
    
    # log the outputs into wandb
    for i in range(min_val_epoch+1):
        wandb.log({'tr_loss' : tr_loss_list[i], 'tr_acc' : tr_acc_list[i], 'val_loss' : val_loss_list[i], 'val_acc' : val_acc_list[i]})
    print('Saving the best model...')
    torch.save(best_model_state, best_model_path)
    print('Best model saved.')
    return
#---------------------------------------------------------------------------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Input Hyperparameters')
    parser.add_argument('-wp'   , '--wandb_project'  , type = str  , default = 'dl_ass_3_script', metavar = '')
    parser.add_argument('-we'   , '--wandb_entity'   , type = str  , default = 'cs22m059', metavar = '')
    parser.add_argument('-d'    , '--dataset'        , type = str  , default = 'hin', metavar = '', choices = languages)
    parser.add_argument('-ct'   , '--cell_type'      , type = str  , default = 'LSTM', metavar = '', choices = ["LSTM", "GRU", "RNN"])
    parser.add_argument('-em'   , '--embedding_size' , type = int  , default = 128, metavar = '')
    parser.add_argument('-hd'   , '--hidden_size'    , type = int  , default = 256, metavar = '')
    parser.add_argument('-en'   , '--enc_num_layers' , type = int  , default = 3, metavar = '')
    parser.add_argument('-dn'   , '--dec_num_layers' , type = int  , default = 3, metavar = '')
    parser.add_argument('-dp'   , '--dropout'        , type = float, default = 0.2, metavar = '')
    parser.add_argument('-bi'   , '--bidirectional'  , type = str  , default = 'Yes', metavar = '', choices = ["Yes", "No"])

    # get the parameters from command line argument into a dictionary
    params = vars(parser.parse_args())
    #--------------------------------------------
    # initialize wandb with given params
    wandb.init(project = params['wandb_project'], config = params)
    print("Provided hyperparameters = ", params)
    params = SimpleNamespace(**params)

    # datapaths
    train_data_path = '/aksharantar_sampled/' + params.dataset + '/' + params.dataset + '_train.csv'
    valid_data_path = '/aksharantar_sampled/' + params.dataset + '/' + params.dataset + '_valid.csv'
    test_data_path = '/aksharantar_sampled/' + params.dataset + '/' + params.dataset + '_test.csv'

    # loads data from datapaths
    train_data = pd.read_csv(train_data_path, sep=',', header=None).values
    test_data = pd.read_csv(test_data_path, sep=',', header=None).values
    valid_data = pd.read_csv(valid_data_path, sep=',', header=None).values

    
    # build vocabulary
    x_vocab, y_vocab = prepareVocab(train_data)

    
    print('Number of characters in Source Vocab :', x_vocab.n_chars-4)
    print('Number of characters in Target Vocab :', y_vocab.n_chars-4)

    # make tensors out of words with padding
    x_train = processData(train_data[:,0], x_vocab, eos=True).to(device=device)
    x_test = processData(test_data[:,0], x_vocab, eos=True).to(device=device)
    x_valid = processData(valid_data[:,0], x_vocab, eos=True).to(device=device)

    y_train = processData(train_data[:,1], y_vocab, sos=True, eos=True).to(device=device)
    y_test = processData(test_data[:,1], y_vocab, sos=True, eos=True).to(device=device)
    y_valid = processData(valid_data[:,1], y_vocab, sos=True, eos=True).to(device=device)

    
    n_train = x_train.size(0)
    n_valid = x_valid.size(0)
    n_test = x_test.size(0)
    print('Number of Training Sequences :', n_train)
    print('Number of Validation Sequences :', n_valid)
    print('Number of Test Sequences :', n_test)

    # convert dataset into dataloader that will help in processing training batch wise
    print("Building the model...")
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(x_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    num_epochs = 15
    learning_rate = 0.001

    # Fixed parameters for encoder and decoder
    input_size_encoder = x_vocab.n_chars
    input_size_decoder = y_vocab.n_chars
    output_size = input_size_decoder

    # Model hyperparameters
    cell_type = params.cell_type
    embedding_size = params.embedding_size
    hidden_size = params.hidden_size  # Needs to be the same for both RNN's
    enc_num_layers = params.enc_num_layers
    dec_num_layers = params.dec_num_layers
    dropout = params.dropout
    bidirectional = True if params.bidirectional == "Yes" else False

    encoder_net = Encoder(
    cell_type, input_size_encoder, embedding_size, hidden_size, enc_num_layers, dropout, bidirectional).to(device)

    decoder_net = Decoder(
        cell_type,
        input_size_decoder,
        embedding_size,
        hidden_size,
        output_size,
        dec_num_layers,
        dropout,
    ).to(device)

    model = Seq2Seq(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # the function saves the best model
    trainModel(model, criterion, optimizer, train_dataloader, valid_dataloader, num_epochs, batch_size)

    # load the best model saved
    model.load_state_dict(torch.load(best_model_path))

    # report test accuracy using the best model
    test_data_df = pd.read_csv(test_data_path, sep=',', header=None)
    test_loss, test_acc = saveAndEvaluate(model, test_dataloader, criterion, test_data_df, y_vocab, batch_size)
    print(f"Test Loss: {test_loss:.2f}")
    print(f"Test Accuracy: {test_acc:.2f}")
    wandb.finish()
