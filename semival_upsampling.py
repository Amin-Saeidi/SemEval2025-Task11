
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,AutoModel, AdamW
import warnings
import matplotlib.pyplot as plt
from sklearn.utils import resample

warnings.filterwarnings('ignore')

dtype_dict = {
    'anger': 'int',
    'disgust': 'int',
    'fear': 'int',
    'joy': 'int',
    'sadness': 'int',
    'surprise': 'int'
}

df_train = pd.read_csv("amh_final_train_translated.csv", dtype=dtype_dict)
df_dev = pd.read_csv("translated_amh_dev_a.csv", dtype=dtype_dict)

df_train = df_train.rename(columns = {'Text':'text'})
df_train['text'] = df_train['text'].fillna('')
df_dev['text'] = df_dev['text'].fillna('')

df_train.drop(['Unnamed: 0'], axis=1, inplace=True)
df_dev.drop(['Unnamed: 0'], axis=1, inplace=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_LEN = 180
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 11
LEARNING_RATE = 1e-5

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")

target_cols = [col for col in df_train.columns if col not in ['text', 'id', 'emotion', 'length_of_class', 'Unnamed: 0', 'translated_text', 'Unnamed: 0.1']]

class SemivalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.max_len = max_len
        self.text = df.text
        self.tokenizer = tokenizer
        self.targets = df[target_cols].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def loss_fn(outputs, targets):
     return torch.nn.BCEWithLogitsLoss(pos_weight=weights_tensor)(outputs, targets)

print("Before upsampling......")
print(df_train['anger'].value_counts())
print(df_train['disgust'].value_counts())
print(df_train['fear'].value_counts())
print(df_train['joy'].value_counts())
print(df_train['sadness'].value_counts())
print(df_train['surprise'].value_counts())


# Define the labels
labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
df_train['label_combination'] = df_train[['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']].astype(str).agg('-'.join, axis=1)

# Find the max count for a minority class (adjust based on need)
fear_count = df_train['fear'].sum()
surprise_count = df_train['surprise'].sum()
joy_count = df_train['joy'].sum()
max_target_size = max(fear_count, surprise_count, joy_count) * 3  # Upsample minorities 3x

# Upsample only the minority classes
df_resampled = pd.concat([
    resample(group, replace=True, n_samples=min(len(group) * 3, max_target_size), random_state=42) 
    if group[labels].sum().max() < 1000 else group 
    for _, group in df_train.groupby('label_combination')
])

# Drop the extra column
df_resampled = df_resampled.drop(columns=['label_combination']).reset_index(drop=True)

print("After upsampling......")
print(df_resampled['anger'].value_counts())
print(df_resampled['disgust'].value_counts())
print(df_resampled['fear'].value_counts())
print(df_resampled['joy'].value_counts())
print(df_resampled['sadness'].value_counts())
print(df_resampled['surprise'].value_counts())

# Compute class weights
class_counts = df_resampled[labels].sum().values
total_samples = len(df_resampled)
weights = total_samples / (6 * class_counts)  # Inverse frequency method

# Convert weights to tensor
weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)


train_dataset = SemivalDataset(df_resampled, tokenizer, MAX_LEN)
valid_dataset = SemivalDataset(df_dev, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True)

class SemivalClassificationClass(torch.nn.Module):
   def __init__(self):
       super(BERTClass, self).__init__()
       self.roberta = AutoModel.from_pretrained("sentence-transformers/LaBSE")
       #self.roberta = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base")
       #self.roberta = AutoModel.from_pretrained("abdulmunimjemal/xlm-r-retrieval-am-v1")
       #self.roberta = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")        
       self.l2 = torch.nn.Dropout(0.4)
       self.fc = torch.nn.Linear(768,6)
    
   def forward(self, ids, mask, token_type_ids):
       _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
       features = self.l2(features)
       output = self.fc(features)
       return output
        
model = SemivalClassificationClass()
model.to(device)
optimizer = AdamW(params =  model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)


def validation():
    model.eval()
    fin_targets = []
    fin_outputs = []
    val_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
      for _, data in enumerate(valid_loader, 0):
          ids = data['ids'].to(device, dtype=torch.long)
          mask = data['mask'].to(device, dtype=torch.long)
          token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
          targets = data['targets'].to(device, dtype=torch.float)
          
          outputs = model(ids, mask, token_type_ids)
          loss = loss_fn(outputs, targets)  # Compute loss
          
          val_loss += loss.item()
          total_batches += 1
          
          fin_targets.extend(targets.cpu().detach().numpy().tolist())
          fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    avg_val_loss = val_loss / total_batches  # Compute average loss

    return avg_val_loss, fin_outputs, fin_targets

def train():
  train_losses = []
  val_losses = []
  
  for epoch in range(EPOCHS):
      model.train()
      epoch_loss = 0.0
      total_batches = 0
      
      for _, data in enumerate(train_loader, 0):
          ids = data['ids'].to(device, dtype=torch.long)
          mask = data['mask'].to(device, dtype=torch.long)
          token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
          targets = data['targets'].to(device, dtype=torch.float)
          
          outputs = model(ids, mask, token_type_ids)
          loss = loss_fn(outputs, targets)
          
          epoch_loss += loss.item()
          total_batches += 1
  
          if _ % 500 == 0:
              print(f'Epoch {epoch+1}, Step {_}, Training Loss: {loss.item()}')
          
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
      
      avg_train_loss = epoch_loss / total_batches
      avg_val_loss, _, _ = validation()
      
      train_losses.append(avg_train_loss)
      val_losses.append(avg_val_loss)
      
      print(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")
  
  return train_losses, val_losses
    
train_losses, val_losses = train()



avg_val_loss, outputs, targets = validation()
outputs = np.array(outputs) >= 0.5
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")

test_file_path = "translated_amh_test_a.csv" 
df_test = pd.read_csv(test_file_path)
df_test['text'] = df_test['text'].fillna('')

class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.text = df.text
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
        
        
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
# Save the plot as an image file (e.g., PNG)
plt.savefig("training_validation_loss.png", dpi=300, bbox_inches='tight')
# Optionally, you can close the figure to free memory
plt.close()


test_dataset = TestDataset(df_test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

def predict():
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            outputs = torch.sigmoid(outputs).cpu().detach().numpy() 
            predictions.extend(outputs)
    return np.array(predictions)

torch.save(model.state_dict(), "sentiment_model.pth")
test_predictions = predict()

binary_predictions = (test_predictions >= 0.5).astype(int)

for i, emotion in enumerate(['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']):
    df_test[emotion] = binary_predictions[:, i]

output_file_path = "translated_amh_test_a.csv"  
df_test.to_csv(output_file_path, index=False, encoding='utf-8-sig')
print(f"Predictions saved to {output_file_path}")