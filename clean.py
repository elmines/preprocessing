"""
Provides utility functions for cleaning text.
"""

import re
import spacy

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
	text = re.sub(r"can not", "cannot", text) #Quirk unique to SpaCy tokenizer
	return text


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
	pass

if __name__ == "__main__":
	sequences = ['Ethan "loves" preprocessing']
	for seq in sequences:
		print(pre_clean_seq(seq))
