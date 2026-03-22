# Adversarial ML Password Model


This repository contains the first draft of a (kind of) trained GPT-2 LLM model that generates passwords from a word data set that contains the top 10000 most used English words. I decided on just a small data set of words at first to target the first part of our dataset design, weak passwords that can be found in the dictionary with no special characters. Works with a-z A-Z and 0-9. Has a very basic training file as well as a script that generates passwords with GPT-2 and runs through zxcvbn.


## words.txt
A dataset of the top 10000 most used English words taken from https://github.com/david47k/top-english-wordlists?tab=readme-ov-file

## generate_dataset.py
Creates the training passwords for GPT-2. Just 100 at first, but we can obviously scale it up when needed. Doesn't need to be run everytime you want to generate passwords with GPT-2, it's just for training the llm. 

## password_dataset.txt
Passwords created by generate_dataset.py. For GPT-2 training only, not running through password meter.

## train_gpt2.py
Tweaks/fine tunes the GPT-2 model.

## generate_passwords.py
This is the main file that generates passwords with GPT-2 and runs it through zxcvbn. 

## Sequence of running files

Run in this order:

``python3 generate_dataset.py``

``python3 train_gpt2.py``

``python3 generate_passwords.py``
