import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,AutoModel, AdamW
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

warnings.filterwarnings('ignore')


dtype_dict = {
    'anger': 'int',
    'disgust': 'int',
    'fear': 'int',
    'joy': 'int',
    'sadness': 'int',
    'surprise': 'int'
}

#df_train = pd.read_csv("amh_final_train_translated.csv", dtype=dtype_dict)

df_train = pd.read_csv("amharic_deepseek_final_train_df.csv", dtype=dtype_dict)
df_dev = pd.read_csv("translated_amh_dev_a.csv", dtype=dtype_dict)

df_train = df_train.rename(columns = {'Text':'text'})
df_train['text'] = df_train['text'].fillna('')
df_dev['text'] = df_dev['text'].fillna('')

def enforce_int(df):

    emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    for column in emotion_columns:
        df[column] = df[column].astype(int)
    return df


df_train = enforce_int(df_train)
df_dev = enforce_int(df_dev)

def get_emotions(row):
    emotions = []
    for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
        if row[emotion] == 1:
            emotions.append(emotion)
    return emotions

    
print(df_train['fear'])   
df_train.drop(['Unnamed: 0'], axis=1, inplace=True)
df_dev.drop(['Unnamed: 0'], axis=1, inplace=True)


df_train = df_train.reset_index(drop=True)
df_dev = df_dev.reset_index(drop=True)
#print(list(df_train['disgust']))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_LEN = 180
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 11
LEARNING_RATE = 1e-5

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
#tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
#tokenizer = AutoTokenizer.from_pretrained("abdulmunimjemal/xlm-r-retrieval-am-v1")
#tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


target_cols = [col for col in df_train.columns if col not in ['text', 'id', 'Unnamed: 0', 'Unnamed: 0.1']]

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


train_dataset = SemivalDataset(df_train, tokenizer, MAX_LEN)
valid_dataset = SemivalDataset(df_dev, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, 
                          num_workers=4, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, 
                        num_workers=4, shuffle=False, pin_memory=True)


class SemivalClassificationClass(torch.nn.Module):
   def __init__(self):
       super(SemivalClassificationClass, self).__init__()
       self.roberta = AutoModel.from_pretrained("sentence-transformers/LaBSE")
       #self.roberta = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base")
       #self.roberta = AutoModel.from_pretrained("abdulmunimjemal/xlm-r-retrieval-am-v1")
       #self.roberta = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")        
       self.l2 = torch.nn.Dropout(0.5)
       self.fc = torch.nn.Linear(768,6)
    
   def forward(self, ids, mask, token_type_ids):
       _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
       features = self.l2(features)
       output = self.fc(features)
       return output


model = SemivalClassificationClass()
model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
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
              print(f'Epoch {epoch+1}/{EPOCHS}, Step {_}, Training Loss: {loss.item()}')
          
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
      
      avg_train_loss = epoch_loss / total_batches
      avg_val_loss, _, _ = validation()
      
      train_losses.append(avg_train_loss)
      val_losses.append(avg_val_loss)
      
      print(f"Epoch {epoch+1}/{EPOCHS}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")
  
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


test_file_path = "amh_test.csv"
df_test = pd.read_csv(test_file_path)
df_test['text'] = df_test['text'].fillna('')




plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig("training_validation_finall_loss.png", dpi=300, bbox_inches='tight')
plt.close()

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

test_dataset = SemivalDataset(df_test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)


def predict():
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


outputs, targets = predict()
outputs = np.array(outputs) >= 0.5
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print(f"test Accuracy Score = {accuracy}")
print(f"test F1 Score (Micro) = {f1_score_micro}")
print(f"test F1 Score (Macro) = {f1_score_macro}")


conf_matrices = multilabel_confusion_matrix(targets, outputs)
print(conf_matrices)
emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

# Plot the confusion matrix for each label and save the images
for i, emotion in enumerate(emotions):
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrices[i], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'], 
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix for {emotion}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.savefig(f'confusion_matrix_{emotion}.png', bbox_inches='tight')  # Save as PNG file
    plt.close()  

# for i, emotion in enumerate(['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']):
#     df_test[emotion] = binary_predictions[:, i]

# output_file_path = "translated_amh_test_a.csv"  
# df_test.to_csv(output_file_path, index=False, encoding='utf-8-sig')
# print(f"Predictions saved to {output_file_path}")