"""
Module for generating a test set based on the VAD values of the text
"""
import pandas as pd
import sys
import warnings

def make_testset(category,df,dialogues,responses,cat_type):
    """
    Extracts responses of a given emotional type and intensity from a dataset

    :param str              category: Strength of emotion (may be either \"high\" or \"low\")
    :param pd.DataFrame           df: Dataframe containing words and their corresponding VAD values
    :param list(list(str)) dialogues: The tokenized prompts, such that dialogues[i][j] is the jth token of the ith prompt
    :param list(list(str)) responses: The tokenized responses, such that dialogues[i][j] is the jth token of the ith response
    :param str              cat_type: The type of emotion (may be one of  \"valence\", \"arousal\", or \"dominance\"

    :returns: A tuple (df_testset, indices) where indices are the indices of the testset responses in the original df
    :rtype:   tuple(pd.DataFrame,list(int))
    """
    if category.lower() == "high":
        if cat_type != "valence":
            df = df[:20]
            df['Word'] = df['Word'].str.lower()
            words = list(set(df['Word'].tolist()))
        else:
            df_poswords = df[:20]
            df_negwords = df[-5000:]
            df_poswords['Word'] = df_poswords['Word'].str.lower()
            df_negwords['Word'] = df_negwords['Word'].str.lower()
            pos_words = list(set(df_poswords['Word'].tolist()))
            pos_words.remove("honest")
            neg_words = list(set(df_negwords['Word'].tolist()))
    else:
        df = df[-20:]
        df['Word'] = df['Word'].str.lower()
        words = list(set(df['Word'].tolist()))
    
    if 'words' in locals():
        if "honest" in words:
            words.remove("honest")
    cat_type = category+ " " + cat_type
    target_questions = []
    target_responses = []
    category_type = []
    indexes = [] # need these for removing them from the original list
    for i, dialogue in enumerate(dialogues):
        switch = False
        for word in dialogue:
            if cat_type != "high valence":
                if word in words:
                    switch = True
                elif word == "honest":
                    switch = False
                    break
            else:
                if word in pos_words:
                    switch = True
                elif word =="honest":
                    switch = False
                    break
                elif switch == True and word in neg_words:
                    switch = False
                    break
        if switch == True:
            target_questions.append(' '.join(dialogues[i]))
            target_responses.append(' '.join(responses[i]))
            category_type.append(cat_type)
            indexes.append(i)
    df_new = pd.DataFrame(index=range(len(target_questions)))
    df_new['target_questions'] = pd.DataFrame(target_questions)
    df_new['target_responses'] = pd.DataFrame(target_responses)
    df_new['category_type'] = pd.DataFrame(category_type)
    df_new['indexes'] = pd.DataFrame(indexes)
    return df_new,indexes

def exclude_testset(dialogues, responses, test_set_path="./test_set_removed.xlsx", regenerate=True, verbose=True):
	"""
	Excludes certain sequences from dialogues and responses based on VAD values and write the sequences to a spreadsheet

	:param list(list(str))     dialogues: The initial dialogues, where each element is a token of a dialogue
	:param list(list(str))     responses: The responses to each dialogue
	:param str             test_set_path: The path to write the spreadsheet
	:param bool               regenerate: If False, don't rewrite the test set spreadsheet; just return the remaining indices
	:param bool                  verbose: Print helpful messages to stderr

	:returns: The indices of the dialogues (and their corresponding responses) that are left over
	:rtype:   list(int)
	"""

	df = pd.read_excel('./Warriner, Kuperman, Brysbaert - 2013 BRM-ANEW expanded.xlsx')
	df = df[['Word','V.Mean.Sum','A.Mean.Sum','D.Mean.Sum']]
	df_valence = df[['Word','V.Mean.Sum']]
	df_valence = df_valence.sort_values(by=['V.Mean.Sum'],ascending=False).reset_index(drop=True)
	df_arousal = df[['Word','A.Mean.Sum']]
	df_arousal = df_arousal.sort_values(by=['A.Mean.Sum'],ascending=False).reset_index(drop=True)
	df_dominance = df[['Word','D.Mean.Sum']]
	df_dominance = df_dominance.sort_values(by=['D.Mean.Sum'],ascending=False).reset_index(drop=True)
	
	warnings.filterwarnings('ignore')
	high_valence, hv_indexes =  make_testset("high",df_valence,dialogues,responses,"valence")
	low_valence, lv_indexes =  make_testset("low",df_valence,dialogues,responses,"valence")
	#------------------------#
	high_arousal, ha_indexes =  make_testset("high",df_arousal,dialogues,responses,"arousal")
	low_arousal, la_indexes =  make_testset("low",df_arousal,dialogues,responses,"arousal")
	#------------------------#
	high_dominance, hd_indexes =  make_testset("high",df_dominance,dialogues,responses,"dominance")
	low_dominance, ld_indexes =  make_testset("low",df_dominance,dialogues,responses,"dominance")

	if regenerate:
		df_new = pd.concat([high_valence, low_valence, high_arousal, low_arousal, high_dominance,low_dominance]).reset_index(drop=True)
		df_new.to_excel(test_set_path, index=False)
		if verbose:
			sys.stderr.write("Wrote test set to {}.\n".format(test_set_path))
	elif verbose:
		sys.stderr.write("Skipped writing test set to spreadsheet.\n")

	test_indices = hv_indexes + lv_indexes + ha_indexes + la_indexes + hd_indexes + ld_indexes
	unique_test_indices = set(test_indices)
	remaining_indices = list( set(range(len(dialogues))) - unique_test_indices )
	remaining_indices.sort()


	return remaining_indices

def test_indices(testset_path):
	"""
	Returns the indices of the original dataset from which the testset exchanges were extracted	

	:param str testset_path: The path to the testset spreadsheet

	:returns: The indices
	:rtype:   list(int)
	"""
	df = pd.read_excel(testset_path)	
	indices = set( df["indexes"] )
	return indices
