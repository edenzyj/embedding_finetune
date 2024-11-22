from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW

import os
import torch
import csv

# Set device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model and tokenizer
model_name = "dunzhang/stella_en_1.5B_v5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Define LoRA configuration
lora_config = LoraConfig(
    r=32,                       # Rank for the low-rank matrices
    lora_alpha=32,             # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Modules to inject LoRA into (e.g., query and value in attention)
    lora_dropout=0.1,          # Dropout for LoRA layers
    bias="none"                # Specify if LoRA should have bias terms
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

csv_file = "6pdfs_queries.csv"

with open(csv_file, 'r') as fr:
    csv_reader = csv.DictReader(fr)
    paragraphs = []
    queries = []
    for row in csv_reader:
        if len(row['txt']) > 4000:
            print("----- size too big -----")
            print(len(row['txt']))
            continue
        for i in range(1, 7):
            paragraphs.append(row['txt'])
            queries.append(row['q{}'.format(i)])
    fr.close()

pairs = [(queries[i], paragraphs[i]) for i in range(len(paragraphs))]

# Create a custom Dataset class
class TextPairDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, context = self.pairs[idx]
        return query, context

# Custom collate function to handle padding within a batch
def collate_fn(batch):
    queries, contexts = zip(*batch)
    queries = tokenizer(list(queries), padding=True, return_tensors="pt", truncation=True)
    contexts = tokenizer(list(contexts), padding=True, return_tensors="pt", truncation=True)
    
    # Move inputs to the selected device (GPU or CPU)
    queries = {key: val.to(device) for key, val in queries.items()}
    contexts = {key: val.to(device) for key, val in contexts.items()}
    return queries, contexts

train_dataset = TextPairDataset(pairs, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=1e-4)

model.train()

finetuned_path = "./results/finetuned_stella1.5B_6pdfs"
if not os.path.isdir(finetuned_path):
    os.mkdir(finetuned_path)

epoch_num = 5
step = 64
for epoch in range(epoch_num):
    print("+++++ Start to finetune! +++++")
    total_loss = 0

    for i, (query, paragraph) in enumerate(train_dataloader):
        query_embed = model(**query).last_hidden_state.mean(dim=1)
        paragraph_embed = model(**paragraph).last_hidden_state.mean(dim=1)

        # Compute loss using cosine similarity
        similarity = cosine_similarity(query_embed, paragraph_embed)
        loss = 1 - similarity.mean()
        loss = loss / step
        total_loss += loss
        loss.backward()

        if (i + 1) % 64 == 0:
            optimizer.zero_grad()
            optimizer.step()
            total_loss = 0

    step_path = finetuned_path + "/epoch_{}".format(epoch)
    if not os.path.isdir(step_path):
        os.mkdir(step_path)
    
    model.save_pretrained(step_path)
    tokenizer.save_pretrained(step_path)

finetuned_path = finetuned_path + "/final"
if not os.path.isdir(finetuned_path):
    os.mkdir(finetuned_path)

model.save_pretrained(finetuned_path)
tokenizer.save_pretrained(finetuned_path)
