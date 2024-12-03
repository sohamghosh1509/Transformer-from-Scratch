from torch.utils.data import DataLoader
from train import word_to_index, index_to_word, vocab, Transformer
from utils import clean_and_tokenize, create_data, TextDataset, calculate_bleu
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


with open('/kaggle/input/corpus-en-fr/dev.en', 'r', encoding='utf-8') as f:
    src_dev_text = f.read()
with open('/kaggle/input/corpus-en-fr/dev.fr', 'r', encoding='utf-8') as f:
    tgt_dev_text = f.read()
with open('/kaggle/input/corpus-en-fr/test.en', 'r', encoding='utf-8') as f:
    src_test_text = f.read()
with open('/kaggle/input/corpus-en-fr/test.fr', 'r', encoding='utf-8') as f:
    tgt_test_text = f.read()

src_tokenized_dev = clean_and_tokenize(src_dev_text)
tgt_tokenized_dev = clean_and_tokenize(tgt_dev_text)
src_tokenized_test = clean_and_tokenize(src_test_text)
tgt_tokenized_test = clean_and_tokenize(tgt_test_text)

src_dev_indices = create_data(src_tokenized_dev, word_to_index)
src_test_indices = create_data(src_tokenized_test, word_to_index)
tgt_dev_indices = create_data(tgt_tokenized_dev, word_to_index)
tgt_test_indices = create_data(tgt_tokenized_test, word_to_index)

dev_dataset = TextDataset(src_dev_indices, tgt_dev_indices)
test_dataset = TextDataset(src_test_indices, tgt_test_indices)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=True, collate_fn=dev_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=test_dataset.collate_fn)

model_load_path = "transformer.pt"
embed_dim = 512
max_seq_len = 500
hidden_dim = 2048
criterion = nn.CrossEntropyLoss()
num_epochs = 10

vocab_size = len(vocab)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(embed_dim, vocab_size, vocab_size, max_seq_len, hidden_dim).to(device)
model.load_state_dict(torch.load(model_load_path))
print(f"Model loaded from {model_load_path}")

optimizer = optim.Adam(model.parameters(), lr=0.0005)

model.eval()
total_loss = 0
total_correct = 0 
total_samples = 0 

with torch.no_grad():
    for inputs, targets in tqdm(test_loader):

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

accuracy = 100.0 * total_correct / total_samples
print(f"Loss: {total_loss / len(test_loader)}, Test_Accuracy: {accuracy:.4f}")

bleu_score, bleu_scores = calculate_bleu(model, vocab, src_tokenized_test, tgt_tokenized_test, index_to_word)
print(f"BLEU_Score: {bleu_score:.4f}")