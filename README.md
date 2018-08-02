# preprocessing
Simple Python package for cleaning text for machine translation, dialog generation, etc.

## Dependencies
- Python 3
- [gensim](https://pypi.org/project/gensim/) (for training Word2Vec embeddings)
- [numpy](https://pypi.org/project/numpy/) (for shuffling data)
- [pandas](https://pypi.org/project/pandas/)
  - [xlrd](https://pypi.org/project/xlrd/) 
- [spacy](https://pypi.org/project/spacy/) (for tokenization)
  - [en\_core\_web\_sm](https://spacy.io/usage/models) language model
- [sphinx](https://pypi.org/project/Sphinx/) 1.7.6 or higher (for generating documentation)
- The spreadsheet of 14,000 English lemmas with their VAD values authored by [Warriner et al.](https://link.springer.com/article/10.3758/s13428-012-0314-x). See spreadsheet [here](https://github.com/elmines/EmotChatbot/blob/master/Warriner%2C%20Kuperman%2C%20Brysbaert%20-%202013%20BRM-ANEW%20expanded.xlsx)
