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

# load and resample to 16k Hz, convert to mono
resampled_audio = {}  # create empty dict: filepath -> Tensor (1D, float32): (1D audio tensor)
for i, path in enumerate(audio_files, 1):
    try:
        waveform, sample_rate = torchaudio.load(path)  # waveform shape: (channels, samples) -> e.g., (2, 44100) for stereo, sr: sample rate (e.g., 16000 Hz)
        # # average all channels into 1 channel (mono) if needed
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   
        # resample to 16k Hz if necessary
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000

        # squeeze channel dimension  (channels, samples) to 1D tensor (samples,)  
        resampled_audio[path] = waveform.squeeze(0)
    except Exception as e:
        print(f"Failed to load/resample {path}: {e}")
    if i % 100 == 0:
        print(f"Processed {i}/{len(audio_files)} files")

print(f"Completed: loaded and resampled {len(resampled_audio)} audio files.")

# Print one resampled audio and its properties
if resampled_audio:
    # Get the first audio file path and tensor
    first_path = list(resampled_audio.keys())[0]
    first_audio = resampled_audio[first_path]
    
    print("\n--- First Audio Sample Properties ---")
    print(f"File path: {first_path}")
    print(f"Tensor shape: {first_audio.shape}")  # e.g., (705600,) for mono audio
    print(f"Tensor dtype: {first_audio.dtype}")  # torch.float32
    print(f"Sample rate: 16000 Hz (all resampled to this)")
    print(f"Duration (seconds): {first_audio.shape[0] / 16000:.2f}")
    print(f"First 10 samples: {first_audio[:10]}")

# Load dataset from CSV
csv_path = os.path.expanduser("~/STT_workspace/data_filepath_and_scripts.csv")
dataset = load_dataset("csv", data_files=csv_path, encoding="euc-kr")

# Check the dataset structure
print(dataset["train"][0])
# It should show something like: {'File Path': '/home/narae/STT_workspace/PCM_converted_AUDIO+TEXT_BK/LIST00000_F_YJS00_41_Metropolitan_indoors_00000.wav', 'Text Content': '정하경님'}

# --- Initialize processor ---

# Tokenize transcripts (use processor Wav2Vec2Processor for the kresnik model (it bundles feature_extractor + tokenizer) instead of Wav2Vec2Tokenizer)
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load the Korean pretrained processor (handles audio preprocessing + text tokenization)
processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

# Tokenize transcripts -> produce label ids for CTC
def preprocess_text(batch):
    # batch["transcript"] is a string or list of strings
    # use processor.tokenizer to convert to token ids
    batch["input_ids"] = processor.tokenizer(batch["Text Content"], padding=True, truncation=True).input_ids
    return batch

# --- Split dataset into Train, Eval, Test ---

from datasets import load_dataset, DatasetDict

# Load the dataset 
dataset_full = load_dataset("csv", data_files=csv_path, encoding="euc-kr")

# Train split 
train_temp_split = dataset_full['train'].train_test_split(test_size=0.2, seed=42)

# Split the 20% test set into half validation  and half test 
val_test_split = train_temp_split['test'].train_test_split(test_size=0.5, seed=42)

# Combine them back into a single DatasetDict for easy handling
dataset = DatasetDict({
    'train': train_temp_split['train'],       # 80% of total
    'validation': val_test_split['train'],    # 10% of total
    'test': val_test_split['test']            # 10% of total
})

print(f"Train size: {len(dataset['train'])}")
print(f"Eval size:  {len(dataset['validation'])}")
print(f"Test size:  {len(dataset['test'])}")

# Apply preprocessing to the dataset
# When you call .map() on a DatasetDict, it automatically applies to all 3 splits
dataset = dataset.map(preprocess_text)

# --- Load model and split into training, evaluation, testing data ---

# Load the pretrained model
model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to('cuda')

# Fine-tune model configuration for new vocabulary size (pad_token_id, vocab_size)
model.config.pad_token_id = processor.tokenizer.pad_token_id 
model.config.vocab_size = len(processor.tokenizer)

from torch.utils.data import DataLoader

# Define a custom collator
def data_collator(batch):
    audio_features = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    return {"input_values": audio_features, "labels": labels}

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

