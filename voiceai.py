# -*- coding: utf-8 -*-
"""voiceAI.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bu7uYtQ0PZq8oWGcohrMfKyx5CNtKF2h

# Requirements
"""

pip install transformers

! pip install datasets

! pip install accelerate

pip install git+https://github.com/openai/whisper.git

pip install jiwer

"""# Import the required libraries"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from glob import glob
import pandas as pd
import re
import os
import jiwer

# Importing the folder to google colab
from google.colab import drive
drive.mount('\content\drive')

# Checking if GPU is available in this machine or not.If not available means then going to run in CPU.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Setting floating point precision of 16bit if we running in GPU otherwise we use 32bit
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Assign Openai whisper model to a variable
model_id = "openai/whisper-large-v3"

# Loading the model and datatype,since we running google colab set the low cpu memory usage is to TRUE
model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )

# Move the model to GPU
model.to(device)

# Setting AutoProcessor for tokenization process and feature extraction from the speech signal
processor = AutoProcessor.from_pretrained(model_id)

# Initialize ASR pipeline from tranformer package
asr_pipeline = pipeline("automatic-speech-recognition",
                        model=model,
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                        max_new_tokens=128,
                        chunk_length_s=30,
                        batch_size=16,
                        return_timestamps=True,
                        torch_dtype=torch_dtype,
                        device=device,
                        )

# Transcribe audio file
def get_transcription(filename: str):
    transcription = asr_pipeline(filename)
    return transcription

# Collecting the path details
path=glob("/content/contentdrive/MyDrive/common_voice_test/*")
path = pd.DataFrame(path,columns= ['location'])

pattern = "common_voice_mr_\d{8}.wav"

data={}
for i in range(len(path)):
  audio_file_path=path['location'].iloc[i]
  transcription = get_transcription(audio_file_path)
  result=re.findall(pattern,audio_file_path)
  data[result[0]]=transcription['text']

# Creating a dataframe with transcribed audio details
Transcribe_audio=pd.DataFrame(data,columns=['File_name','Transcribe_audio'])

# Select the refrence audio file path
path="/content/trans.txt"

# Specify the encoding, such as 'utf-8'
with open(path, encoding='utf-8') as file:
    content = file.read()

# Split the string by both newline and tab characters
lines_and_tabs = re.split(r'\n|\t', content)

data={}
# Print each element
for i in range(0,len(lines_and_tabs),2):
    data[lines_and_tabs[i]]=lines_and_tabs[i+1]

for old_key in list(data.keys()):
    new_key = old_key.replace('\ufeff','')
    data[new_key]=data.pop(old_key)

# Creating a dataframe with refrence auido file details
Reference_audio=pd.DataFrame(data.items(),columns=['File_name','Refrence_audio'])

# Combining the two dataframes based on file name
Speech_to_text = pd.merge(Transcribe_audio, Reference_audio, on='ID', how='inner')

"""# Evaluation metrics for the ASR -- Automatic Speech Recognition is (WER) Word Error Rate, Character Error Rate (CER), Sentence Error Rate(SER)"""

def Evauluation_metric(reference,hypothesis):
  wer = jiwer.wer(reference, hypothesis)
  return wer

Speech_to_text['Wer']=Evauluation_metric(Speech_to_text['Refrence_audio'],Speech_to_text['Transcribe_audio'])

Speech_to_text.to_csv('Speech_to_text.tsv', header=True, index=False)