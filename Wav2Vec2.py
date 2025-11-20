# check torch version
print(torch.__version__)
# check that GPU is available
torch.cuda.is_available()

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import torch

processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to('cuda')

# The model will be saved inside this directory
output_dir="./STT_workspace/wav2vec2-korean-medical" 

# ... inside the Trainer loop ...
# The Trainer will automatically save checkpoints like:
# ./wav2vec2-korean-medical/checkpoint-500/



# ---------Evaluation on Zeroth-Korean ASR corpus---------

ds = load_dataset("bingsu/zeroth-korean")
test_ds = ds['test']

print(test_ds[0])

def map_to_pred(batch):
    speech_samples = [item['array'] for item in batch['audio']]
    inputs = processor(speech_samples, sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")
    #attention_mask = inputs.attention_mask.to("cuda")
    
    with torch.no_grad():
        #logits = model(input_values, attention_mask=attention_mask).logits
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = test_ds.map(map_to_pred, batched=True, batch_size=16)

import evaluate as eval
wer = eval.load("wer")
cer = eval.load("cer")
result_wer = wer.compute(references=result["text"], predictions=result["transcription"])
result_cer = cer.compute(references=result["text"], predictions=result["transcription"])
print(f'WER:{result_wer*100:.2f}%')
print(f'CER:{result_cer*100:.2f}%')