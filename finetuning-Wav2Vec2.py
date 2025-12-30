# Install in virtual environment:
# pip3 install transformers datasets torchaudio
# pip install torchcodec

# --- Fine-tuning on our STT dataset ---

import torchaudio
from datasets import load_dataset
import os

print("Current working dir: ",os.getcwd()) 

# dir_path = os.path.dirname(os.path.realpath(__file__)) 
# print("Directory of this python script: ",dir_path)

# Directory to search -> Make sure to use the PCM converted folder!!
search_dir = os.path.expanduser("~/STT_workspace/PCM_converted_AUDIO+TEXT_BK")

# The model will be saved inside this directory
output_dir="~/STT_workspace/wav2vec2-korean-medical" 

# gather all audio .wav file paths
audio_extensions = (".wav")
audio_files = []
for root, _, files in os.walk(search_dir): # root: current directory path, dirs: directories inside root but we don't need this so ignore using _ underscore , files: files inside root
    for fname in files:
        if fname.lower().endswith(audio_extensions): # if the file ends with .wav
            audio_files.append(os.path.join(root, fname))
audio_files.sort()

# Load dataset from CSV
csv_path = os.path.expanduser("~/STT_workspace/data_filepath_and_scripts.csv")
dataset = load_dataset("csv", data_files=csv_path, encoding="euc-kr")

# # Check the dataset structure
# print("\n--- Check data structure ---")
# print(dataset["train"][0],"\n")
# It should show something like: {'File Path': '/home/narae/STT_workspace/PCM_converted_AUDIO+TEXT_BK/LIST00000_F_YJS00_41_Metropolitan_indoors_00000.wav', 'Text Content': '정하경님'}

# --- Initialize processor ---

# Tokenize transcripts (use processor Wav2Vec2Processor for the kresnik model (it bundles feature_extractor + tokenizer) instead of Wav2Vec2Tokenizer)
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load the Korean pretrained processor (handles audio preprocessing + text tokenization)
processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

# Load the pretrained model
model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to('cuda')

# DELETE ----# Fine-tune model configuration for new vocabulary size (pad_token_id, vocab_size)
# model.config.pad_token_id = processor.tokenizer.pad_token_id 
# model.config.vocab_size = len(processor.tokenizer)

# --- Split dataset into Train, Eval, Test ---
from datasets import load_dataset, DatasetDict

# Train split 
train_temp_split = dataset['train'].train_test_split(test_size=0.2, seed=42)

# Split the 20% test set into half validation  and half test 
val_test_split = train_temp_split['test'].train_test_split(test_size=0.5, seed=42)

# Combine them back into a single DatasetDict for easy handling
dataset = DatasetDict({
    'train': train_temp_split['train'],       # 80% 
    'validation': val_test_split['train'],    # 10% 
    'test': val_test_split['test']            # 10% 
})

print(f"Train size: {len(dataset['train'])}")
print(f"Eval size:  {len(dataset['validation'])}")
print(f"Test size:  {len(dataset['test'])}")

# --- Preprocess text and audio into model inputs/labels ---
def prepare_dataset(batch):
    input_values = []
    labels = []
    for file_path, transcript in zip(batch["File_Path"], batch["Text_Content"]):
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1: # make mono if needed
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000: # resample to 16k Hz if needed
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        array = waveform.squeeze().numpy()
        # processor will handle feature extraction (normalization etc.)
        inp = processor(array, sampling_rate=16000).input_values[0] 
        input_values.append(inp)
        # tokenize transcript to label ids
        label_ids = processor.tokenizer(transcript).input_ids
        labels.append(label_ids)

    batch["input_values"] = input_values
    batch["labels"] = labels
    return batch

# apply the mapping (batched=True so mapping sees lists)
dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset["train"].column_names)
print("Audio + text preprocessing done!")


from torch.utils.data import DataLoader
# Define a custom collator: the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.
def data_collator(features):
    # features is a list of dicts with "input_values" (list[float]) and "labels" (list[int])
    input_features = [{"input_values": f["input_values"]} for f in features]
    batch = processor.pad(input_features, return_tensors="pt")

    labels = [f["labels"] for f in features]
    # pad labels (tokenizer.pad returns a dict with 'input_ids')
    padded_labels = processor.tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")["input_ids"]
    # set padding token id to -100 so loss ignores them
    padded_labels[padded_labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = padded_labels
    return batch

# Create DataLoader
train_loader = DataLoader(
    dataset['train'], batch_size=16, shuffle=True, collate_fn=data_collator
)

eval_loader = DataLoader(
    dataset['validation'], batch_size=16, shuffle=False, collate_fn=data_collator
)

test_loader = DataLoader(
    dataset['test'], batch_size=16, shuffle=False, collate_fn=data_collator
)

# --- Training the model ---

# Trainer API for quick prototyping
from transformers import TrainingArguments, Trainer
# pip install accelerate -U
# pip install 'accelerate>=0.26.0'

# Define all the parameters related to training 
training_args = TrainingArguments(
    remove_unused_columns=False,
    output_dir="./results",
    eval_strategy="steps",
    save_steps=500,
    learning_rate=3e-4,
    per_device_train_batch_size=4,     # If you experience a CUDA out-of-memory (OOM) when running the above cell, incrementally reduce the `batch_size` by factors of 2 until you find a batch size that fits your device.
    gradient_accumulation_steps=4,     # Number of updates steps to accumulate the gradients before performing a backward/update pass. Helps stimulate an effective larger batch size (4 gradient_acc_step * 4 batch_size = batch size of 16) than what your GPU memory can actually support
    num_train_epochs=5,
    warmup_steps=500,
    logging_dir="./logs",
    fp16=True,  # Enable mixed precision for faster training on GPUs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
)

# Start training
print("\n--------------------------------")
print("Starting fine-tuning...\n")
trainer.train()


# --- Evaluate the model on test set ---
from evaluate import load

# Load WER metric
wer_metric = load("wer")
# Load CER metric
cer_metric = load("cer")

# Get predictions and references from test set
import torch
predictions = []
references = []

model.eval()
max_examples = 5
with torch.no_grad():
    for batch in test_loader:
        input_values = batch["input_values"].to(model.device)
        logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        
        # Decode predictions
        preds = processor.batch_decode(pred_ids)
        predictions.extend(preds)
        
        # Decode references
        labels = batch["labels"].cpu().numpy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        refs = processor.batch_decode(labels, group_tokens=False)
        references.extend(refs)
        
        if len(predictions) >= max_examples:
            predictions = predictions[:max_examples]
            references = references[:max_examples]
            break

# Compute WER
wer = wer_metric.compute(predictions=predictions, references=references)
print(f"Word Error Rate: {wer:.2f}")
# Compute CER
cer = cer_metric.compute(predictions=predictions, references=references)
print(f"Character Error Rate: {cer:.2f}")