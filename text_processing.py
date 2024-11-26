import pandas as pd
import numpy as np
import re
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from tqdm import tqdm

english = "OpenSubtitles.en-fr.en"
french = "OpenSubtitles.en-fr.fr"

with open(english, 'r', encoding='utf-8') as f:
  eng_lines = f.read().split('\n')

with open(french, 'r', encoding='utf-8') as f:
  fre_lines = f.read().split('\n')

def bullet_removal(corpus):
  cleaned_corpus = [item.lstrip('-').lstrip() if item.startswith('-') else item for item in corpus]
  return cleaned_corpus

eng_cleaned = bullet_removal(eng_lines)
fre_cleaned = bullet_removal(fre_lines)

# english corpus contains spaces after apostrophes
eng_cleaned = [re.sub(r"\'\s", "'", line) for line in eng_cleaned]

eng_cleaned = [re.sub(r"[l]\s", "I ", line) if line.startswith('l ') else line for line in eng_cleaned]

def tagging(corpus):
  
  corpus = ["<START>" + line + "<END>" for line in corpus]

  return corpus

eng = tagging(eng_cleaned)
fre = tagging(fre_cleaned)

input_docs = []
target_docs = []

input_tokens = set()
target_tokens = set()

for count in tqdm(range(len(eng))):

  input_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", eng[count]))
  input_docs.append(input_docs)

  target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", fre[count]))
  target_docs.append(target_doc)

  for token in input_doc:
    input_tokens.add(token)

  for token in target_doc:
    target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

# Create num_encoder_tokens and num_decoder_tokens:
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

# calculate max length to set dimensions for the one-hot encoded matrix
max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])
