"""
Module for postprocessing spreadsheets

Usage: `python sheets raw.xlsx postprocessed.xlsx`
"""
import sys
import os
import pandas as pd



if os.path.basename(sys.argv[0]) in os.listdir(os.path.dirname(os.path.abspath(__file__))):
	import corpus
else:
	from . import corpus

def postprocess(df, cols=None, clean_fn=None):
	"""
	:param pd.DataFrame df: DataFrame with the text to be postprocessed
	:param list(str) cols: Columns of dataframe to clean. If not provided, have the function infers which to clean.
	:param (str) -> (str) clean_fn: Function for postprocessing a single string (default is corpus.post_clean_seq)

	:return The dataframe with the cleaned data
	:rtype pd.DataFrame
	"""

	if cols is None:
		labels = df.columns
		substrings = ["prompt", "question", "input" "response", "answer", "target", "beam", "prediction"]
		cols = []
		for label in labels:
			for substring in substrings:
				if substring in label:
					cols.append(label)
					break
		if len(cols) < 1:
			raise ValueError("Could not find columns to clean--try providing them yourself.")
	if clean_fn is None: clean_fn = corpus.post_clean_seq

	sys.stderr.write("Cleaning columns {}\n".format(cols))

	for col in cols:
		df[col] = df[col].apply(clean_fn)
	return df


if __name__ == "__main__":

	if len(sys.argv) < 3:
		sys.stderr.write("{}\n".format(__doc__))
		sys.exit(0)
	df = pd.read_excel(sys.argv[1])
	cleaned_df = postprocess(df)
	cleaned_df.to_excel(sys.argv[2], index=False)
