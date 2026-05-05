# Adversarial ML Password Model


This repository contains the first draft of a trained LLM model that generates passwords from a word data set that contains the most used passwords in the world. We then modify those passwords and try to score a 3 or above on the zxcvbn strength meter test. After collecting 10k of these "strong" passwords. We train are TinyLLama LLM to make passwords that can fool password strength meters.


## password_words.txt
Uses the most popular passwords in the world GitHub

## generate_dataset.py
Creates the training passwords. Will create passwords till the 10k quota is met.

## train_tinyllama.py
Tweaks/fine tunes the LLM.

## generate_passwords_tinyllama.py
This is the main file that generates passwords with TinyLLama and runs it through zxcvbn. 

## evaluate.py
Will compare are dataset of passwords to are trained LLM passwords and display results.

## Sequence of running files

Run in this order:

``python3 generate_dataset.py``

``python3 train_tinyllama.py``

``python3 generate_passwords_tinyllama.py``

``python3 evaluate.py —generated (whatever the file name is)``
