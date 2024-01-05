# Automatic Speech Recognition

## Objective:

The primary objective of this internship assignment is to evaluate your proficiency in natural
language processing by developing a highly accurate transcription model for the Marathi
language. This entails creating a robust model that can accurately transcribe spoken
Marathi content. You will be provided with a test dataset, and you are encouraged to use
open source datasets to enhance the model's performance and training.

## Goal:

-	The Voice AI project aims to implement a Speech-to-Text system using the Hugging Face Whisper ASR models. 
-	The primary objectives include accurate transcription of Marathi audio and model fine-tuning for improved performance.
- Our goal is to reduce the wer rate and find the average wer rate of the asr model.

 ## Problem Statement:
- Develop a Asr model to predict the speech accurately(marathi language) and convert into to text.(speech to text).

## Methodology:

-	The project utilizes the Hugging Face Whisper small ASR models for automatic speech recognition. Fine-tuning strategies.
- Hugging face : https://huggingface.co/

## Test Dataset:
Data - https://drive.google.com/drive/folders/1MarQPhe-jhKF9ZwZgxh_UoctTVK-h99R

## Evaluation Metrics:

-	Word Error Rate (WER) is employed as the primary metric for evaluating model performance. 
-	The goal is to minimize WER, ensuring accurate transcription of Marathi speech.
-	calculated Average WER is 0.43 for Whisper small.

## Challenges Faced:

Challenges encountered during the project include GPU memory limitations, fine-tuning difficulties, and handling large models. Strategies to overcome these challenges are discussed. 

-	Storage Constraints: The limited storage capacity in Google Colab posed a challenge, preventing the completion of additional fine-tuning steps due to insufficient space for model checkpoints and intermediate results.
-	Low GPU Resources: The free version of Google Colab provided inadequate GPU capacity, hindering the fine-tuning of larger and more complex models. This limitation impacted the training efficiency and overall model performance.
-	Model Complexity vs Steps: Balancing increased model complexity with a lower number of fine-tuning steps presented a challenge. The compromise led to a higher Word Error Rate (WER), indicating the impact of insufficient training steps on the model's language understanding and transcription accuracy.

## Results:

-	Due to storage and GPU limitations, the Voice AI project faced challenges, leading to incomplete fine-tuning, reduced model performance, and trade-offs in model size. These constraints may result in suboptimal transcription accuracy and language understanding	.
-	This Fine tuning was not working as expected. But I tried my best to perform tuning.

## Credits:
- Hugging Face's pretrained Whisper-Small model by saiteja malyala : https://huggingface.co/steja/whisper-small-marathi
- finetune : https://huggingface.co/openai/whisper-large-v3

## Project

Asr model files : https://github.com/PooventhiranM/NLP_assignment.git
