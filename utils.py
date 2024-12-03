import re
from torch.utils.data import DataLoader, Dataset
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
nltk.download('punkt')
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import math
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

def clean_and_tokenize(text):
    sentences = text.strip().split('\n')
#     print(len(sentences))
    tokenized_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s]', '', sentence)  # Remove special characters
        
        tokens = word_tokenize(sentence.lower())  # Tokenize and lowercase
        tokens = ['<START>'] + tokens + ['<END>']
        tokenized_sentences.append(tokens)
            
    return tokenized_sentences

def create_data(tokenized_data, word_to_index):
  word_indices = []
  for sentence in tokenized_data:
    word_indices.append([word_to_index[token] if token in word_to_index else word_to_index['<UNK>'] for token in sentence])
  return word_indices

class TextDataset(Dataset):
    def __init__(self, src_indices, tgt_indices, pad_token='<PAD>', max_seq_len=500):
        self.src_indices = [src for src in src_indices if len(src) <= max_seq_len]
        self.tgt_indices = [tgt for src, tgt in zip(src_indices, tgt_indices) if len(src) <= max_seq_len]
        self.pad_token = pad_token

    def __len__(self):
        return len(self.src_indices)

    def __getitem__(self, index):
        input_sequence = np.array(self.src_indices[index])
        output_sequence = np.array(self.tgt_indices[index])

        return torch.tensor(input_sequence), torch.tensor(output_sequence)

    def collate_fn(self, batch):
        sequences, targets = zip(*batch)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
        return padded_sequences, padded_targets
    
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_positional_encoding(d_model, max_len):
        
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pos_enc = torch.zeros(max_len, d_model)
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc.unsqueeze(0)

def generate_tgt_mask(tgt):
    
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    sz = tgt.size(1)
    causal_mask = (torch.tril(torch.ones(sz, sz)) == 1).bool()
    tgt_mask = causal_mask.to(device) & tgt_mask.to(device)
    return tgt_mask

def generate_src_mask(src):
    
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    return src_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)
       
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  
        
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        seq_length_query = query.size(1)
        
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        
        k = self.key_matrix(key)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
       
        k_adjusted = k.transpose(-1,-2)
        product = torch.matmul(q, k_adjusted)
        
        product = product / math.sqrt(self.single_head_dim)
        
        if mask is not None:
            mask = mask.to(device)
            product = product.masked_fill(mask == 0, -1e9)
        
        scores = F.softmax(product, dim=-1)
 
        scores = torch.matmul(scores, v) 
        
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)
        
        output = self.out(concat)
       
        return output
    
def translate_sentence(model, src_sentence, vocab):

    model.eval()
    
    src_sentence = torch.tensor([vocab[word] if word in vocab else vocab['<UNK>'] for word in src_sentence]).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
    tgt_sentence = torch.tensor([vocab['<START>']]).unsqueeze(0).to(device)  # Start token for decoding

    max_len = 100
    predicted_translation = [vocab['<START>']]
    
    with torch.no_grad():
        for _ in range(max_len):
            output = model(src_sentence, tgt_sentence)
            next_word = output[:, -1, :].argmax(1).item()
#             next_word = output.argmax(2)[:, -1].item()
            predicted_translation.append(next_word)
            
            if next_word == vocab['<END>']:
                break
            
            tgt_sentence = torch.cat([tgt_sentence, torch.tensor([[next_word]], device=device)], dim=1)

    return predicted_translation

def calculate_bleu(model, vocab, src_tokenized, tgt_tokenized, index_to_word):
    
    model.eval()
    
    sum_bleu_scores = 0
    bleu_scores = []

    with torch.no_grad():
        for src_sentence, tgt_sentence in tqdm(zip(src_tokenized, tgt_tokenized), total=len(src_tokenized)):

            predicted_translation = translate_sentence(model, src_sentence, vocab)
            predicted_translation_words = [index_to_word[idx] for idx in predicted_translation]

            reference = [tgt_sentence]
            candidate = predicted_translation_words
            bleu_score = sentence_bleu(reference, candidate)
            sum_bleu_scores += bleu_score
            bleu_scores.append((tgt_sentence, bleu_score))
    
    avg_bleu_score = sum_bleu_scores / len(bleu_scores)
    return avg_bleu_score, bleu_scores

def write_bleu_scores_to_file(output_file, bleu_scores):
    with open(output_file, 'w') as f:
        for tgt_sentence_words, bleu_score in bleu_scores:
            # Convert sentence lists to strings
            tgt_sentence_words = [word for word in tgt_sentence_words if word not in {'<START>', '<END>'}]
            tgt_sentence_str = ' '.join(tgt_sentence_words)
            f.write(f"{tgt_sentence_str} {bleu_score:.4f}\n")

def plot_performance_train(train_losses, title):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, marker='o', color='tab:red', label='Train Loss')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_performance_dev_bleu(dev_losses, bleu_scores):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(dev_losses) + 1), dev_losses, marker='o', label="Dev Loss")
    plt.title("Dev Loss Across Hyperparameter Sets")
    plt.xlabel("Hyperparameter Set")
    plt.ylabel("Loss")
    plt.xticks(range(1, len(dev_losses) + 1), [f"Set {i+1}" for i in range(len(dev_losses))])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(bleu_scores) + 1), bleu_scores, marker='o', label="BLEU Score", color='orange')
    plt.title("BLEU Score Across Hyperparameter Sets")
    plt.xlabel("Hyperparameter Set")
    plt.ylabel("BLEU Score")
    plt.xticks(range(1, len(bleu_scores) + 1), [f"Set {i+1}" for i in range(len(bleu_scores))])
    plt.legend()

    plt.tight_layout()
    plt.show()