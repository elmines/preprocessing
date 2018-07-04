"""
Provides utility functions for cleaning text.
"""
import sys
import os

import re
import spacy
import numpy as np #For shuffling data
np.random.seed(1)

#Local modules
from . import testset


def preprocess(lines_path, conversations_path, min_line_length=1, max_line_length=60, prompts_path="./prompts.txt", answers_path="./answers.txt", verbose=True):
	"""
	Preprocesses the text and generates a test set.
	"""

	with open(lines_path, "r", encoding="utf-8", errors="ignore") as r:
		lines = r.read().split("\n")
	with open(conversations_path, "r", encoding="utf-8", errors="ignore") as r:
		conv_lines = r.read().split("\n")
	if verbose: sys.stderr.write("Read sequences from Cornell files\n")

	(prompts, answers) = generate_conversations(lines, conv_lines)
	if verbose: sys.stderr.write("{} dialog exchanges total.\n".format(len(prompts)))

	prompts = [ pre_clean_seq(prompt).split() for prompt in prompts]
	answers = [ pre_clean_seq(answer).split() for answer in answers]
	if verbose: sys.stderr.write("Cleaned all prompts and answers.\n")
	#prompts, answers are now list(list(str))'s

	(prompts, answers) = filter_by_length(prompts, answers, min_line_length, max_line_length)
	if verbose:
		sys.stderr.write("Filtered out sequences with less than {} or more than {} tokens; {} exchanges remaining.\n".format(min_line_length, max_line_length, len(prompts)))
	
	remaining_indices = testset.exclude_testset(prompts, answers)
	prompts = [prompts[index] for index in remaining_indices]
	answers = [answers[index] for index in remaining_indices]
	if verbose: sys.stderr.write("{} exchanges remaining after generating testset.\n".format(len(prompts)))

	with open(prompts_path, "w", encoding="utf-8") as w:
		w.write( "\n".join( [" ".join(prompt) for prompt in prompts] ) )
	if verbose:
		sys.stderr.write("Wrote non-test prompts to {}\n".format(prompts_path))

	with open(answers_path, "w", encoding="utf-8") as w:
		w.write( "\n".join( [" ".join(answer) for answer in answers] ) )
	if verbose:
		sys.stderr.write("Wrote non-test answers to {}\n".format(answers_path))

def training_sets(prompts_path, answers_path, max_vocab=12000, unk = "<UNK>", output_dir="corpora/", verbose=True):
	"""

	Note that the vocabulary file written to output_dir will always have the unknown token as its first entry
	"""

	with open(prompts_path, "r", encoding="utf-8") as r:
		prompts = [line.split() for line in r.readlines()]
	with open(answers_path, "r", encoding="utf-8") as r:
		answers = [line.split() for line in r.readlines()]

	vocab = [unk] + generate_vocab(prompts, answers, max_vocab)
	vocab2int = {word:index for (index, word) in enumerate(vocab) }
	if verbose: sys.stderr.write("Generated the vocabulary; unknown token is {}:{}\n".format(unk, vocab2int[unk]))
	
	prompts = replace_unknowns(prompts, vocab2int, unk)
	answers = replace_unknowns(answers, vocab2int, unk)
	if verbose: sys.stderr.write("Replaced out-of-vocabulary words with {}.\n".format(unk))

	assert len(prompts) == len(answers)
	shuffled_indices = np.random.permutation(len(prompts))
	prompts = [prompts[index] for index in shuffled_indices]
	answers = [answers[index] for index in shuffled_indices]
	if verbose: sys.stderr.write("Shuffled the dataset.\n")

	train_indices = []
	valid_indices = []
	for i, answer in enumerate(answers):
		if unk in answer:
			valid_indices.append(i)
		else:
			train_indices.append(i)

	for (purpose, indices) in zip( ["train", "valid"], [train_indices, valid_indices] ):
		prompt_lines = [prompts[index] for index in indices]
		prompts_path = os.path.join(output_dir, purpose + "_prompts.txt")
		with open(prompts_path, "w", encoding="utf-8") as r:
			r.write( "\n".join( [" ".join(prompt) for prompt in prompt_lines] ) )
		if verbose: sys.stderr.write("Wrote {} lines to {}.\n".format(len(prompt_lines), prompts_path))

		answer_lines = [answers[index] for index in indices]
		answers_path = os.path.join(output_dir, purpose + "_answers.txt")
		with open(answers_path, "w", encoding="utf-8") as r:
			r.write( "\n".join( [" ".join(answer) for answer in answer_lines] ) )
		if verbose: sys.stderr.write("Wrote {} lines to {}.\n".format(len(answer_lines), answers_path))

	with open("vocab.txt", "w", encoding="utf-8") as w:
		w.write( "\n".join( ["{} {}".format(word, index) for (word, index) in vocab2int.items()] ) )
		
	vocab_path = os.path.join(output_dir, "vocab.txt")
	if verbose: sys.stderr.write("Wrote vocabulary to {}.\n".format(vocab_path))

def generate_conversations(lines, conv_lines):
	# Create a dictionary to map each line's id with its text
	ids = []
	lines_text = []
	for line in lines:
		_line = line.split(' +++$+++ ')
		if len(_line) == 5: #Lines not of length 5 are not properly formatted
			ids.append(_line[0])
			lines_text.append( _line[4] )

	id2line = { id_no:line for (id_no, line) in zip(ids, lines_text) }

	# Create a list of all of the conversations' lines' ids.
	convs = []
	for line in conv_lines[:-1]:
	    	_line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
	    	convs.append(_line.split(','))
	# Sort the sentences into prompts (inputs) and answers (targets)
	prompts = []
	answers = []
	for conv in convs:
		for i in range(len(conv)-1):
			prompts.append(id2line[conv[i]])
			answers.append(id2line[conv[i+1]])
	return (prompts, answers)

def pre_clean_seq(text):
	"""
	Cleans a single text sequence.
	
	:param str text: Text to be cleaned.
	
	:returns The cleaned text
	:rtype str
	"""
	
	text = re.sub("(``)|('')", ' " ', text) #These are already in the corpus and could screw up the tokenizer
	
	text = text.lower()
	
	#Pronoun contractions
	text = re.sub(r"i'm", "i am", text)

	#'s contractions -- can't do a blanket expression without matching possessive 's substrings
	text = re.sub(r"he's", "he is", text)
	text = re.sub(r"she's", "she is", text)
	text = re.sub(r"it's", "it is", text)
	text = re.sub(r"that's", "that is", text)
	text = re.sub(r"what's", "what is", text)
	text = re.sub(r"where's", "where is", text)
	text = re.sub(r"who's", "who is", text)
	text = re.sub(r"how's", "how is", text)

	#Special cases for contractions
	text = re.sub(r"won't", "will not", text)
	text = re.sub(r"can't", "cannot", text)

	#Contraction "suffixes"
	text = re.sub(r"\'ll", " will", text)
	text = re.sub(r"\'ve", " have", text)
	text = re.sub(r"\'re", " are", text)
	text = re.sub(r"\'d", " would", text)
	text = re.sub(r"n't", " not", text)
	text = re.sub(r"n'", "ng", text)
	text = re.sub(r"'bout", "about", text)
	text = re.sub(r"'til", "until", text)

		
	#Punctuation/Symbols
	text = re.sub("&quot;", ' " ', text)         
	text = re.sub("&amp;", ' & ', text)          
	text = re.sub("(<.*?>)|(&.*?;)", "", text)            #HTML tags and entities
	text = re.sub(r'[\?\.\!\-]+(?=[\?\.\!\-])', '', text) #Duplicate end punctuation
	text = re.sub(r"\. \. \.", "...", text)               #Compress ellipses to one token

	text = re.sub('\s+', ' ', text ).strip()             #Replace special whitespace characters with simple spaces

	text = tokenize(text)
	
	return text

#TODO: Join separated contraction tokens (like 'll, 's, and n't) to main words,
#		join "can not", --> "cannot", etc.	
def post_clean_seq(text):
	raise NotImplementedError

_nlp = spacy.load("en_core_web_sm")
_tokenizer = spacy.lang.en.English().Defaults.create_tokenizer(_nlp)
def tokenize(text, single_str=True):
	"""
	Tokenizes a single text sequence.

	:param str text: Text to be cleaned

	:returns The tokenized text
	:rtype str
	"""
	text = " ".join([str(tok) for tok in _tokenizer(text)])
	return text

def filter_by_length(prompts, answers, min_line_length, max_line_length):
	"""
	Returns
		short_prompts - a 2-D list of strings where short_prompts[i][j] is the jth token of the ith sequence
		short_answers - a 2-D list of strings like short_prompts
	"""
	# Filter out the prompts that are too short/long
	short_prompts_temp = []
	short_answers_temp = []
	for (i, prompt) in enumerate(prompts):
		if len(prompt) >= min_line_length and len(prompt) <= max_line_length:
			short_prompts_temp.append(prompt)
			short_answers_temp.append(answers[i])
	# Filter out the answers that are too short/long
	short_prompts = []
	short_answers = []
	for (i, answer) in enumerate(short_answers_temp):
		if len(answer) >= min_line_length and len(answer) <= max_line_length:
	        	short_answers.append(answer)
	        	short_prompts.append(short_prompts_temp[i])
	return (short_prompts, short_answers)

def generate_vocab(prompts, answers, max_vocab):
	"""
	prompts - A 2-D list of strings
	answers - A 2-D list of strings
	"""
	word_freq = {}
	for prompt in prompts:
    		for word in prompt:
        		if word not in word_freq: word_freq[word]  = 1
        		else:                     word_freq[word] += 1
	for answer in answers:
    		for word in answer:
        		if word not in word_freq: word_freq[word]  = 1
        		else:                     word_freq[word] += 1

	sorted_by_freq = sorted(word_freq.keys(), key=lambda word: word_freq[word], reverse=True)
	del word_freq

	if len(sorted_by_freq) < max_vocab:
		vocab = sorted_by_freq
	else:
		vocab = sorted_by_freq[:max_vocab]
	return vocab

def replace_unknowns(sequences, vocab, unk):
	"""
	sequences - A 2-D list of strings
	Returns
		A 2-D list of strings
	"""
	return [ [word if word in vocab else unk for word in sequence] for sequence in sequences ]

