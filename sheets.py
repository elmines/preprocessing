"""
Module/script for postprocessing spreadsheets
"""
import sys
import os
import argparse
import re

import pandas as pd

#Local modules
if os.path.basename(sys.argv[0]) in os.listdir(os.path.dirname(os.path.abspath(__file__))):
	import corpus
else:
	from . import corpus

def create_parser():
	"""
	Generates the parser for parsing command-line arguments

	:returns: The parser
	:rtype: argparse.ArgumentParser
	"""
	parser = argparse.ArgumentParser(description="Apply a transformation to a dialogue spreadsheet produced by testest.py, EmotChatbot, etc.")

	parser.add_argument("--input", "-i", metavar="in.xlsx", required=True, help="Path to spreadsheet to transform")
	parser.add_argument("--output", "-o", metavar="out.xlsx", default="out.xlsx", help="Path to outputted spreadsheet (default \"out.xlsx\")")

	parser.add_argument("--postprocess", "--post", action="store_true", help="Apply postprocessing to text")
	parser.add_argument("--no-punct", action="store_true", help="Remove all punctuation marks from text")
	parser.add_argument("--unique", "-u", action="store_true", help="Exclude duplicate prompts")

	return parser


def default_text_cols(df):
	"""
	Attempts to find columns in a dataframe that probably contain text sequences

	:param pd.DataFrame df: The dataframe

	:returns: A list of columns in df that may have text
	:rtype:   list(str)
	"""
	labels = df.columns
	substrings = ["prompt", "question", "input" "response", "answer", "target", "beam", "prediction"]
	cols = []
	for label in labels:
		for substring in substrings:
			if substring in label:
				cols.append(label)
				break
	return cols

def postprocess(df, cols=None, clean_fn=None):
	"""
	:param pd.DataFrame       df: DataFrame with the text to be postprocessed
	:param list(str)        cols: Columns of dataframe to clean. If not provided, have the function infers which to clean.
	:param callable     clean_fn: Postprocessing function with signature clean_fn(str) -> str (default is :py:func:`corpus.post_clean_seq`)

	:return: The dataframe with the cleaned data
	:rtype:  pd.DataFrame
	"""

	if cols is None: cols = default_text_cols(df)
	if len(cols) < 1: raise ValueError("Could not find columns to clean--try providing them yourself.")
	sys.stderr.write("Cleaning columns {}\n".format(cols))

	if clean_fn is None: clean_fn = corpus.post_clean_seq
	for col in cols:
		df[col] = df[col].apply(clean_fn)

	return df


def remove_punct(df, cols=None):
	"""
	:param pd.DataFrame       df: DataFrame with the text to be postprocessed
	:param list(str)        cols: Columns of dataframe to clean. If not provided, have the function infers which to clean.

	:return: The dataframe with the cleaned data
	:rtype:  pd.DataFrame
	"""
	return postprocess(df, cols=cols, clean_fn=_punct_filters)	

def trim(df, col=None):
	"""
	:param pd.DataFrame df: DataFrame with text to be processed
	:param str          col: Column used to identify duplicate rows (default \"indexes\")
	
	:returns: The dataframe with the filtered data
	:rtype: pd.DataFrame
	"""

	if col is None: col = "indexes"
	keys = []
	indices = []
	for i, key in enumerate(df[col]):
		if key not in keys:
			keys.append(key)
			indices.append(i)
	df_trimmed = df.iloc[indices]
	return df_trimmed

def _punct_filters(text):
	"""
	:param str text: Single string to clean

	:returns: text with all punctuation removed
	:rtype: str
	"""
	#Don't sub apostrophes, lest we ruin contractions
	text = re.sub(r"[\.\?\!,;:]", " ", text)
	text = re.sub(r"\s+", " ", text)
	text = text.strip()
	return text

if __name__ == "__main__":
	parser = create_parser()
	args = parser.parse_args()

	transformations = [args.postprocess, args.no_punct, args.unique]
	if sum(transformations) != 1:
		print("Must specify exactly one transformation: --postprocess, --unique")
		sys.exit(0)

	df = pd.read_excel(args.input)
	if args.postprocess:	
		df_trans = postprocess(df)
	elif args.no_punct:
		df_trans = remove_punct(df)
	elif args.unique:
		df_trans = trim(df)

	df_trans.to_excel(args.output, index=False)
