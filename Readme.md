# Baseline

## Install

Install libraries

- NLTK
- NLTK (punkt)
- NLTK (averaged_perceptron_tagger)
- NLTK (stem) 
	- in python terminal type....
	import nltk
	nltk.download('wordnet')
- NLTK (words)
-scikit-learn

## Usage

Open the jupyter notebook `Main.ipynb` and just execute the code cell by cell.
The output file is by default `output.json`

- To change the train file change the file to open in the 3rd cell from `development.json`
- to change output file change the file to open in the 4th cell from `output.json`

If you want to run our program without using jupyter notebook, run the individual python files (i.e. baseline1.py, baseline2.py, baseline3.py, etc.) to generate individual output files. Then, pass in these files as arguments for evaluate.py, in addition to the type of training json.