# transformers - Access to a large collection of pre-trained models for various NLP tasks
# datasets - simplify the process of downloading, preparing, and using datasets
# accelerate - distributed training support , designed to accelerate training and inference on GPUs and TPUs.
# jiwer - provides metrics to evaluate ASR systems

!pip install transformers  datasets accelerate
!pip install jiwer

from google.colab import drive
drive.mount('/content/drive')

# imported necessary modules
# torch used for tensor operations and neural networks.
# AutoModelForSpeechSeq2Seq loads pre-trained speech-to-sequence models.
# AutoProcessor loads pre-trained tokenizers/processors associated with specific models.
# pipeline simplifies the use of pre-trained models for specific NLP tasks, like text speech2text
# datasets library for loading and working with datasets commonly used in machine learning tasks
# pandas data manipulation purpose
# Numpy used for calculations
# jiwer library for calculating Word Error Rate (WER) and other metrics used in evaluating ASR systems
# regular expression module pattern matching in string

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import pandas as pd
import numpy as np
import jiwer
import re
import os

# Avoids unwanted errors
import warnings
warnings.filterwarnings("ignore")


# Transcribe audio file function
def get_transcription(filename: str):
  
  # checks Gpu is available or not
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # OpenAI whisper small model pretrained on marathi language by saiteja malyala
  model="steja/whisper-small-marathi"

  # Setting AutoProcessor for tokenization process and feature extraction from the speech signal
  processor = AutoProcessor.from_pretrained("steja/whisper-small-marathi")

  model = AutoModelForSpeechSeq2Seq.from_pretrained("steja/whisper-small-marathi")

  model.to(device)

  # Initialize ASR pipeline from tranformer package
  pipe = pipeline(
      # Specifies the task for the pipeline.
      "automatic-speech-recognition",
      # assign the whisper model to ASR pipeline.
      model=model,
      # Uses the tokenizer from the initialized processor.
      tokenizer=processor.tokenizer,
      # utilizes the feature extractor from the processor.
      feature_extractor=processor.feature_extractor,
      # Sets the maximum number of new tokens.
      max_new_tokens=128,
      # Defines the length of audio chunks in seconds.
      chunk_length_s=30,
      # Specifies the batch size for ASR processing.
      batch_size=16,
      # Sets the device (GPU or CPU) for processing.
      device=device,
      )
  
  transcription = pipe(filename,return_timestamps=False)

  return transcription

def drop_special_chars(text):
    # Defines a regular expression pattern that includes a set of characters to be matched and removed
    pattern = '[,\|\?\.\!\-\;\:\"\“\%\‘\”\�।]'                                  
     # Uses the re.sub function to substitute all occurrences of the characters matched by the pattern with an empty string in the given text.
    cleaned_text = re.sub(pattern, '', text).strip()                           
    return cleaned_text.strip() 

# This function returns a list containing the names of the entries in the specified directory.
file_path = os.listdir('/content/drive/MyDrive/common_voice_test')
print(len(file_path))

# All these details are stored in the result_ dictionary, providing comprehensive information about each processed audio file
result_ = { 'audio' : [] , 'Transcribe_text' : [] }
result_

for i in range(len(file_path)):
  if file_path[i] != "trans.txt":
    # The code processes a list of audio files (file_path) using automatic speech recognition (pipe).
    result_ = get_transcription(f'/content/drive/MyDrive/common_voice_test/{file_path[i]}')
    # For each audio file, it extracts information such as the file name, generated text (after removing special characters) from the first chunk.
    result_['audio'].append(file_path[i])
    result_['Transcribe_text'].append(drop_special_chars(result_['text']))

# Converts the data_ dictionary into dataframe
Transcribe_data = pd.DataFrame(result_)                                               
Transcribe_data

# Imports reference details in table format
Reference_data = pd.read_table('/content/drive/MyDrive/common_voice_test/trans.txt',delimiter='\t',names=['audio','text'])
Reference_data.head()

 # merges two DataFrames, Transcribe_data and Reference_data, on the 'audio' column using an inner join.
merged_df = pd.merge(Transcribe_data,Reference_data, on='audio', how='inner')             
merged_df

word_error_rate = []

# This code calculates the Word Error Rate (WER) for each pair of generated text ('generated_text') and actual text ('text') in the merged_df DataFrame.
# It iterates through each row, retrieves the generated and actual texts, computes the WER using the jiwer library, and appends the WER (multiplied by 100 for percentage) to the 'word_error_rate' list
for i in range(merged_df.shape[0]):
  hyp,ref = merged_df.loc[i,['generated_text','text']].values                   
  wer = jiwer.wer(hyp,ref)                                                     
  word_error_rate.append(wer*100)

# converrting into a column
merged_df['wer'] = word_error_rate
merged_df

# Gives the average of wer
np.mean(merged_df['wer'])

# save the df as csv file
merged_df.to_csv('Asr_model_results.csv',index=False)