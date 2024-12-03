from collections import Counter
from torch.utils.data import DataLoader
from utils import clean_and_tokenize, create_data, TextDataset, generate_src_mask, generate_tgt_mask, calculate_bleu, plot_performance_train, plot_performance_dev_bleu, write_bleu_scores_to_file
from test import dev_loader, src_tokenized_test, tgt_tokenized_test
from encoder import TransformerEncoder
from decoder import TransformerDecoder
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math

with open('/kaggle/input/corpus-en-fr/train.en', 'r', encoding='utf-8') as f:
    src_train_text = f.read()
with open('/kaggle/input/corpus-en-fr/train.fr', 'r', encoding='utf-8') as f:
    tgt_train_text = f.read()

src_tokenized_train = clean_and_tokenize(src_train_text)
tgt_tokenized_train = clean_and_tokenize(tgt_train_text)

word_freq = {}
for sentence in src_tokenized_train:
    for word in sentence:
        word_freq[word] = word_freq.get(word, 0) + 1

for sentence in tgt_tokenized_train:
    for word in sentence:
        word_freq[word] = word_freq.get(word, 0) + 1

for i, sentence in enumerate(src_tokenized_train):
    src_tokenized_train[i] = [word if word_freq[word] >= 3 else '<UNK>' for word in sentence]
    
for i, sentence in enumerate(tgt_tokenized_train):
    tgt_tokenized_train[i] = [word if word_freq[word] >= 3 else '<UNK>' for word in sentence]

vocab = Counter()
vocab['<PAD>'] = 0
i = 1
for sentence in src_tokenized_train:
    for word in sentence:
        if word not in vocab:
          vocab[word] = i
          i += 1
            
for sentence in tgt_tokenized_train:
    for word in sentence:
        if word not in vocab:
          vocab[word] = i
          i += 1
            
vocab_size = len(vocab)
word_to_index = {word: i for i, (word, _) in enumerate(vocab.items())}
index_to_word = {i: word for i, (word, _) in enumerate(vocab.items())}

src_train_indices = create_data(src_tokenized_train, word_to_index)
tgt_train_indices = create_data(tgt_tokenized_train, word_to_index)

train_dataset = TextDataset(src_train_indices, tgt_train_indices)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)

class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, max_seq_len, hidden_dim, num_layers=6, n_heads=8, dropout=0.2):
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(src_vocab_size, embed_dim, max_seq_len, hidden_dim, num_layers=num_layers, n_heads=n_heads, dropout=dropout)
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, max_seq_len, hidden_dim, num_layers=num_layers, n_heads=n_heads, dropout=dropout)
        
    def forward(self, src, tgt):
        src_mask = generate_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        tgt_mask = generate_tgt_mask(tgt)
        out = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        return out
    
embed_dim = 512
max_seq_len = 500
hidden_dim = 2048
criterion = nn.CrossEntropyLoss()
num_epochs = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, optimizer, train_loader):
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for inputs, targets in tqdm(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()            
            outputs = model(inputs, targets[:, :-1])
            
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets[:, 1:].contiguous().view(-1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            loss.backward()
            optimizer.step()

        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * total_correct / total_samples
        
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}, Train_Accuracy: {accuracy:.4f}")
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
    return train_losses, train_accuracies

def evaluate_model(model, optimizer, data_loader):

    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
        # Forward pass
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, targets[:, :-1])

            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets[:, 1:].contiguous().view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * total_correct / total_samples
        
    return avg_loss, accuracy 

hyperparameter_sets = [
    {"num_layers": 2, "n_heads": 8, "embed_dim": 512, "dropout": 0.1},
    {"num_layers": 4, "n_heads": 8, "embed_dim": 512, "dropout": 0.2},
    {"num_layers": 4, "n_heads": 12, "embed_dim": 768, "dropout": 0.2},
    {"num_layers": 6, "n_heads": 12, "embed_dim": 768, "dropout": 0.1}
]

dev_losses_list = []
bleu_scores_list = []
train_losses_list = []

best_bleu_score = 0
best_bleu_scores = []
best_model = None

for i, params in enumerate(hyperparameter_sets):
    print(f"Training with Hyperparameter Set {i + 1}: {params}")
    model = Transformer(embed_dim=params["embed_dim"],
                        src_vocab_size=vocab_size,
                        target_vocab_size=vocab_size,
                        max_seq_len=max_seq_len,
                        hidden_dim=hidden_dim,
                        num_layers=params["num_layers"],
                        n_heads=params["n_heads"],
                        dropout=params["dropout"]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_losses, train_accuracies = train_model(model, optimizer, train_loader)
    dev_loss, dev_accuracy = evaluate_model(model, optimizer, dev_loader)
    print(f"Loss: {dev_loss}, Dev_Accuracy: {dev_accuracy:.4f}") 
    bleu_score, bleu_scores = calculate_bleu(model, vocab, src_tokenized_test, tgt_tokenized_test, index_to_word)
    print(f"BLEU_Score: {bleu_score:.4f}")
    
    if bleu_score > best_bleu_score:
        best_bleu_score = bleu_score
        best_bleu_scores = bleu_scores
        best_model = model
    
    dev_losses_list.append(dev_loss)
    bleu_scores_list.append(bleu_score)
    train_losses_list.append(train_losses)

model_save_path = "transformer.pt"
torch.save(best_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

output_file = "testbleu.txt"
write_bleu_scores_to_file(output_file, best_bleu_scores)

for i in range(4):
    plot_performance_train(train_losses_list[i], f"Hyperparameter Set {i + 1}")
    
plot_performance_dev_bleu(dev_losses_list, bleu_scores_list)