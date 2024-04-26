# Import necessary libraries
import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
import spacy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traceback
import warnings
from collections import Counter, defaultdict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab, build_vocab_from_iterator

# Configure logging
logging.basicConfig(filename='training_logs.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load the spaCy model for tokenization
nlp = spacy.load("en_core_web_sm")

# Function to load data from multiple file paths and save the combined data
def load_and_save_data(file_paths, save_path):
    data_frames = [pd.read_excel(path) for path in file_paths]
    combined_data = pd.concat(data_frames)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined_data.to_excel(save_path, index=False)
   
    return combined_data

# Encoder class definition
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=n_layers, dropout=(dropout if n_layers > 1 else 0))

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)  
        return outputs, hidden, cell

# Attention mechanism class definition
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim + dec_hid_dim), dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0) 

        src_len = encoder_outputs.size(0)
        hidden = hidden.repeat(src_len, 1, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        hidden = hidden.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)
    
# Decoder class definition with attention
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim + emb_dim), dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim + dec_hid_dim + emb_dim), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, cell):
        input = input.unsqueeze(0)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))
        embedded = embedded.squeeze(1)  
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        assert (output == hidden).all(), "Output and Hidden state are not equal."

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden, cell

# Seq2Seq model class definition
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, encoder_outputs, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

def tokenizer(text):
    return [token.text for token in nlp(str(text))]

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Function to tokenize a given text using the spaCy NLP model loaded previously.
def tokenize_text(text):
    return [token.text for token in nlp(text)]

# Function to tokenize all datasets (training, validation, test) and return a combined list of tokenized texts.
def tokenize_all_data(train_data, validate_data, test_data):
    all_tokenized_texts = []
    all_tokenized_texts.extend(tokenize_data(train_data))
    all_tokenized_texts.extend(tokenize_data(validate_data))
    all_tokenized_texts.extend(tokenize_data(test_data))
    return all_tokenized_texts

# Helper function to tokenize a dataset.
def tokenize_data(data):
    tokenized_texts = [tokenize_text(str(text)) for text in data['Value.log.text']]
    return tokenized_texts

# Builds a vocabulary from a list of tokenized texts. 
def build_vocab_from_tokenized_texts(tokenized_texts):
    counter = Counter(token for text in tokenized_texts for token in text)
    vocabulary = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
    vocabulary.update({token: i+4 for i, (token, _) in enumerate(counter.items())})
    return vocabulary

# Constructs a vocabulary from a collection of token frequencies.
def build_vocabulary(counter):
    vocabulary = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocabulary.set_default_index(vocabulary['<unk>'])
    return vocabulary

# Convert a sequence of token IDs back to their corresponding words using the vocabulary.
def ids_to_words(ids, vocab):
    if isinstance(ids, int):
        return [vocab.itos[ids]]
    elif hasattr(ids, '__iter__'):
        return [vocab.itos[id] if id < len(vocab.itos) else '<unk>' for id in ids]
    else:
        return ['<unk>']

def yield_raw_texts_from_dataframes(*dataframes):
    for dataframe in dataframes:
        for text in dataframe['Value.log.text']:
            yield text

# Adapt yield_tokens to use raw_texts_iterator
def yield_tokenized_texts(raw_texts_iterator):
    for text in raw_texts_iterator:
        yield tokenizer(text)

# Collate function for DataLoader.
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_batch, trg_batch

# Function to load datasets into DataLoader objects, enabling efficient batch processing during model training and evaluation.
def load_datasets(train_file, val_file, test_file, tokenizer, max_seq_length, vocabulary, batch_size):
    train_data = CustomDataset(train_file, tokenizer, vocabulary, max_seq_length)
    valid_data = CustomDataset(val_file, tokenizer, vocabulary, max_seq_length)
    test_data = CustomDataset(test_file, tokenizer, vocabulary, max_seq_length)
    test_data_loop = CustomDataset(test_file, tokenizer, vocabulary, max_seq_length)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    test_loader_loop = DataLoader(test_data_loop, batch_size=batch_size)   
    
    return train_loader, valid_loader, test_loader, test_loader_loop

def get_data_from_row(row, text_files_dir):
    if row['Text vs Voice'] == 'Text':
        return row['Value.log.text']
    elif row['Text vs Voice'] == 'Voice':
        file_path = os.path.join(text_files_dir, row['Steps'] + '.txt')
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            # print(f"File not found: {file_path}. Skipping row.")
            return "" 

class CustomDataset(Dataset):
    def __init__(self, file_paths, max_seq_length, text_files_dir, vocabulary, include_name=False, include_turn_count=False):
        self.dataframes = [pd.read_excel(file) for file in file_paths]
        self.data = pd.concat(self.dataframes, ignore_index=True)
        self.max_seq_length = max_seq_length
        self.text_files_dir = text_files_dir
        self.vocab_size = len(vocabulary)
        self.data['Value.log.text'] = self.data['Value.log.text'].astype(str)
        self.include_name = include_name
        self.include_turn_count = include_turn_count 

        # Initialize and fit Keras Tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.data['Value.log.text']) 
        self.vocab_size = len(self.tokenizer.word_index) + 1  
        self.data['turn_count'] = self.data.apply(lambda row: self.determine_turn_count(row), axis=1)

    def determine_turn_count(self, row):
        return len(row['Value.log.text'].split('\n'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_data = get_data_from_row(row, self.text_files_dir)
        tokenized_text = self.tokenizer.texts_to_sequences([text_data])
        padded_text = pad_sequences(tokenized_text, maxlen=self.max_seq_length, padding='post')[0]
        text_tensor = torch.tensor(padded_text, dtype=torch.long)

        if self.include_name and self.include_turn_count:
            name = row['Name']
            turn_count = self.determine_turn_count(row)
            return text_tensor, text_tensor, name, turn_count
        elif self.include_name:
            name = row['Name']
            return text_tensor, text_tensor, name
        else:
            return text_tensor, text_tensor
    
    def calculate_turns_from_name(self, name):
        conversation_rows = self.data[self.data['Name'] == name]
        turn_count = len(conversation_rows) // 2 
        return turn_count

# Calculate the BLEU score for flat lists of predictions and true values, converting IDs to tokens using the vocabulary.
def calculate_bleu_for_flat_preds(preds, true, global_vocab):
    index_to_token = global_vocab.get_itos()
    scores = []

    for pred_id, true_id in zip(preds, true):
        pred_token = [index_to_token[pred_id] if pred_id < len(index_to_token) else '<unk>']
        true_token = [[index_to_token[true_id]] if true_id < len(index_to_token) else ['<unk>']]
        score = sentence_bleu(true_token, pred_token, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0

# Compute accuracy, precision, recall, F1 score, and BLEU score for predictions against true labels, translating IDs to tokens.
def calculate_metrics(preds, true, global_vocab):
    chencherry = SmoothingFunction()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()
    
    itos = global_vocab.get_itos()
    true_tokens = [[itos[i] if i < len(itos) else '<unk>' for i in true]]
    pred_tokens = [itos[i] if i < len(itos) else '<unk>' for i in preds]
    
    bleu_score_avg = sentence_bleu(true_tokens, pred_tokens, smoothing_function=chencherry.method1)
    accuracy = accuracy_score(true, preds)
    precision = precision_score(true, preds, average='weighted', zero_division=0)
    recall = recall_score(true, preds, average='weighted', zero_division=0)
    f1 = f1_score(true, preds, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1, bleu_score_avg

def process_function(model, iterator, optimizer, criterion, device, CLIP, global_vocab, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    all_preds = []
    all_trues = []

    with torch.set_grad_enabled(is_train):
        for _, batch in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)

            optimizer.zero_grad()
            output = model(src, trg)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()

            preds = torch.argmax(output, dim=1).cpu().numpy()
            trues = trg.cpu().numpy()

            all_preds.extend(preds)
            all_trues.extend(trues)

    accuracy, precision, recall, f1, bleu = calculate_metrics(np.array(all_preds), np.array(all_trues), global_vocab)
    
    return epoch_loss / len(iterator), accuracy, precision, recall, f1, bleu

# Train, Validate, and Test Loop
def train_validate_test_loop(model, train_loader, valid_loader, test_loader, optimizer, criterion, device, n_epochs, scheduler=None):
    for epoch in range(n_epochs):
        # Training
        train_loss, train_accuracy, train_precision, train_recall, train_f1, train_bleu = process_function(model, train_loader, criterion, device, is_train=True)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}, BLEU: {train_bleu:.4f}")

        # Validation
        valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1, valid_bleu = process_function(model, valid_loader, criterion, device, is_train=False)
        print(f"Epoch: {epoch+1}, Valid Loss: {valid_loss:.4f}, Acc: {valid_accuracy:.4f}, Prec: {valid_precision:.4f}, Rec: {valid_recall:.4f}, F1: {valid_f1:.4f}, BLEU: {valid_bleu:.4f}")

        if scheduler:
            scheduler.step(valid_loss)

    # Test
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_bleu = process_function(model, test_loader, criterion, device, is_train=False)
    print(f"Test Loss: {test_loss:.4f}, Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}, BLEU: {test_bleu:.4f}")

def test_loop(model, test_loader_loop, criterion, device, global_vocab):
    model.eval()
    conversation_metrics = defaultdict(lambda: {'predictions': [], 'true_labels': []})
    conversation_line_counts = defaultdict(int)

    with torch.no_grad():
        for inputs, targets, names, _ in test_loader_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, targets, 0)
            _, predicted_indices = torch.max(outputs, 2)

            for i, name in enumerate(names):
                conversation_metrics[name]['predictions'].extend(predicted_indices[i].tolist())
                conversation_metrics[name]['true_labels'].extend(targets[i].tolist())
                conversation_line_counts[name] += 1

    line_count_metrics = defaultdict(lambda: {'metrics': [], 'conversation_count': 0})
    for name, data in conversation_metrics.items():
        accuracy, precision, recall, f1, bleu = calculate_metrics(data['predictions'], data['true_labels'], global_vocab)
        metrics = [accuracy, precision, recall, f1, bleu]
        line_count = conversation_line_counts[name]
        line_count_metrics[line_count]['metrics'].append(metrics)
        line_count_metrics[line_count]['conversation_count'] += 1

    avg_grouped_metrics = {}
    for line_count, data in line_count_metrics.items():
        metrics = np.array(data['metrics'])
        avg_metrics = metrics.mean(axis=0) 
        conversation_count = data['conversation_count']
        avg_grouped_metrics[line_count] = {'avg_metrics': avg_metrics.tolist(), 'conversation_count': conversation_count}

    return avg_grouped_metrics

# Save the computed evaluation metrics to a file for analysis.
def write_results(epoch, phase, metrics_str, round_name):
    directory = 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Downloads/RNN/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, f"{round_name}_{phase.lower()}_results_epoch_{epoch+1}.txt")
    try:
        with open(file_path, 'a') as file:
            file.write(metrics_str + '\n')
    except Exception as e:
        print(f"Failed to write {phase} metrics: {e}")

# Calculate and return the elapsed time of an operation in minutes and seconds.
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_train_results(epoch, loss, accuracy, precision, recall, f1, bleu):
    metrics_str = f"Epoch {epoch + 1} - Training: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, BLEU: {bleu:.4f}"
    print(metrics_str)
    return metrics_str

def print_eval_results(epoch, loss, accuracy, precision, recall, f1, bleu, phase='Validation'):
    metrics_str = f"Epoch {epoch + 1} - {phase}: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, BLEU: {bleu:.4f}"
    print(metrics_str)
    return metrics_str

def print_test_results(loss, accuracy, precision, recall, f1, bleu):
    metrics_str = f"Testing: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, BLEU: {bleu:.4f}"
    print(metrics_str)
    return metrics_str

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Orchestrates the complete workflow for processing audio data, training, validating, and testing a sequence-to-sequence model with attention mechanism on rounds of data.
def main():
    # Directories being used
    text_files_dir = r'C:\Users\jaco\OneDrive\Desktop\PHD\Python\MP3_Files\Text'
    model_save_directory = r'C:\Users\jaco\OneDrive\Desktop\PHD\Python\Downloads\RNN'
    os.makedirs(model_save_directory, exist_ok=True) 

    # Define model parameters
    CLIP = 1
    dropout = 0.5
    emb_dim = 256
    hid_dim = 512
    learning_rate = 0.0005
    n_epochs = 10
    n_layers = 1
    vocabulary = {} 
    max_seq_length = 100 
    train_loss = validate_loss = 0 
    batch_size = 32
    patience = 4
    patience_counter = 0
    teacher_forcing_ratio=0.5
  
   # Defines a structured collection of data rounds for sequential processing. 
    rounds = [
            {
                'name': 'Round1_Group1',
                'train': ['C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group1_Train.xlsx'],
                'validate': ['C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group1_Validate.xlsx'],
                'test': ['C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group1_Test.xlsx']
            },
            {
                'name': 'Round2_Group1_and_Group2',
                'train': ['C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group1_Train.xlsx', 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group2_Train.xlsx'],
                'validate': ['C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group1_Validate.xlsx', 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group2_Validate.xlsx'],
                'test': ['C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group1_Test.xlsx', 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group2_Test.xlsx']
            },
            {
                'name': 'Round3_Group1_Group2_and_Group3',
                'train': ['C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group1_Train.xlsx', 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group2_Train.xlsx', 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group3_Train.xlsx'],
                'validate': ['C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group1_Validate.xlsx', 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group2_Validate.xlsx', 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group3_Validate.xlsx'],
                'test': ['C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group1_Test.xlsx', 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group2_Test.xlsx', 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files/Group3_Test.xlsx']
            }
        ]

    # Loop over each round
    for round_data in rounds:
        round_name = round_data['name']
        best_val_loss = float('inf')
        model_save_path = os.path.join(model_save_directory, f'{round_name}_best_model.pt')
        
        train_files = round_data['train']
        validate_files = round_data['validate']
        test_files = round_data['test']
        
        print(f"Starting round: {round_name}")
        start_time = time.time()

        # Define file paths for saving combined data
        train_save_path = f'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Downloads/RNN/{round_name}_train_combined.xlsx'
        validate_save_path = f'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Downloads/RNN/{round_name}_validate_combined.xlsx'
        test_save_path = f'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Downloads/RNN/{round_name}_test_combined.xlsx'

        # Load, combine, and save training data
        train_data = load_and_save_data(round_data['train'], train_save_path)
        print(f"Number of lines in training data for {round_name}: {len(train_data)}")

        # Load, combine, and save validation data
        validate_data = load_and_save_data(round_data['validate'], validate_save_path)
        print(f"Number of lines in validation data for {round_name}: {len(validate_data)}")

        # Load, combine, and save test data
        test_data = load_and_save_data(round_data['test'], test_save_path)
        print(f"Number of lines in test data for {round_name}: {len(test_data)}") 

        # Tokenize data
        tokenized_data_path = 'tokenized_data.pkl'
        if os.path.exists(tokenized_data_path):
            with open(tokenized_data_path, 'rb') as file:
                all_tokenized_texts = pickle.load(file)
        else:
            all_tokenized_texts = tokenize_all_data(train_data, validate_data, test_data)
            with open(tokenized_data_path, 'wb') as file:
                pickle.dump(all_tokenized_texts, file)

        # Build vocabulary
        vocabulary_path = 'vocabulary.pkl'
        if os.path.exists(vocabulary_path):
            with open(vocabulary_path, 'rb') as file:
                vocabulary = pickle.load(file)
        else:
            vocabulary = build_vocab_from_tokenized_texts(all_tokenized_texts)
            with open(vocabulary_path, 'wb') as file:
                pickle.dump(vocabulary, file)
        
        tokenized_texts = tokenize_all_data(train_data, validate_data, test_data)
        vocabulary = build_vocab_from_tokenized_texts(tokenized_texts)

        raw_texts_iterator = yield_raw_texts_from_dataframes(train_data, validate_data, test_data)
        tokenized_texts_iterator = yield_tokenized_texts(raw_texts_iterator)
        global_vocab = build_vocab_from_iterator(tokenized_texts_iterator, specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        global_vocab.set_default_index(global_vocab["<unk>"])
        data_str = ""

        # Create DataLoader instances
        train_loader = DataLoader(CustomDataset(train_files, max_seq_length, text_files_dir, vocabulary), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(CustomDataset(validate_files, max_seq_length, text_files_dir, vocabulary), batch_size=batch_size)
        test_loader = DataLoader(CustomDataset(test_files, max_seq_length, text_files_dir, vocabulary), batch_size=batch_size)
        test_loader_loop = DataLoader(CustomDataset(test_files, max_seq_length, text_files_dir, vocabulary, include_name=True, include_turn_count=True), batch_size=batch_size)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = len(vocabulary) 
        output_dim = len(vocabulary)  

        encoder = Encoder(input_dim=input_dim, emb_dim=emb_dim, enc_hid_dim=hid_dim, dropout=dropout, n_layers=n_layers).to(device)
        attention = BahdanauAttention(enc_hid_dim=hid_dim, dec_hid_dim=hid_dim).to(device)
        decoder = DecoderWithAttention(output_dim=output_dim, emb_dim=emb_dim, enc_hid_dim=hid_dim, dec_hid_dim=hid_dim, dropout=dropout, attention=attention).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if '<pad>' not in vocabulary:
            raise ValueError("'<pad>' token is missing in the vocabulary.")
        pad_token_idx = vocabulary['<pad>']  
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.1, verbose=True)
        
        def find_max_token_id(data_loader):
            max_token_id = 0
            for batch in data_loader:
                input_sequences = batch[0]
                current_max = input_sequences.max()
                if current_max > max_token_id:
                    max_token_id = current_max
            return max_token_id.item()

        # Apply the function to your datasets
        max_id_train = find_max_token_id(train_loader)
        max_id_valid = find_max_token_id(valid_loader)
        max_id_test = find_max_token_id(test_loader)

        # Training and Validation
        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss, train_accuracy, train_precision, train_recall, train_f1, train_bleu = process_function(model, train_loader, optimizer, criterion, device, CLIP, global_vocab, is_train=True)
            train_results_str = print_train_results(epoch, train_loss, train_accuracy, train_precision, train_recall, train_f1, train_bleu)
            write_results(epoch, 'training', train_results_str, round_name)

            # Validation
            model.eval()
            valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1, valid_bleu = process_function(model, valid_loader, optimizer, criterion, device, CLIP, global_vocab, is_train=False)
            eval_results_str = print_eval_results(epoch, valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1, valid_bleu)
            write_results(epoch, 'validation', eval_results_str, round_name)

            # Update best validation loss and reset patience counter if current validation loss is lower
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                patience_counter = 0 
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Stopping early due to no improvement in validation loss.")
                    break

            if scheduler:
                scheduler.step(valid_loss)

        # Test with the best model
        model.load_state_dict(torch.load(model_save_path))
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_bleu = process_function(model, test_loader, optimizer, criterion, device, CLIP, global_vocab, is_train=False)
        test_results_str = print_test_results(test_loss, test_accuracy, test_precision, test_recall, test_f1, test_bleu)
        test_save_path = f'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Downloads/RNN/{round_name}_testing_results.txt'
        with open(test_save_path, 'a') as file:
            file.write(test_results_str + '\n')

        # Test with the loop
        model.load_state_dict(torch.load(model_save_path))
        avg_grouped_metrics = test_loop(model, test_loader_loop, criterion, device, global_vocab)
        test_save_path = f'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Downloads/RNN/{round_name}_global_test_results.txt'
        for line_count, data in sorted(avg_grouped_metrics.items()):
            metrics_str = f"Group (Number of conversation lines: {line_count}, Conversations: {data['conversation_count']})"
            avg_metrics = data['avg_metrics']
            metrics_str += f", Accuracy: {avg_metrics[0]:.4f}, Precision: {avg_metrics[1]:.4f}, Recall: {avg_metrics[2]:.4f}, F1: {avg_metrics[3]:.4f}, Bleu: {avg_metrics[4]:.4f}\n"
            data_str += metrics_str

        data_str += "\n"
        print(data_str)

        with open(test_save_path, 'a') as file:
            file.write(data_str)     
        
        end_time = time.time()
        print(f"Round {round_data['name']} completed in {(end_time - start_time):.2f} seconds.\n")

if __name__ == "__main__":
    main()
