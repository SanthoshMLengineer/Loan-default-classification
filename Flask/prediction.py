from os.path import abspath

import pickle

import numpy as np

# Constants
num_outcome = {0:"NO",
			   1:"YES"
			   }

outcome_display = {"YES":"You are eligible for loan you can apply",
				   "NO":"""You have not eligible for 
				   loan decrease loan amount or increase loan term"""
				   }


def load_model():
	"""
	This function helps to load model
	Parameter
	---------
		None
	Returns
	-------
		model = object
			loaded model
	"""
	try:
		model = pickle.load(open(".//..//Data Files//Random_forest.pkl",
			'rb'))
		return model
	except Exception as e:
		raise e
	

def predict(data_input):
	"""
	This function helps to do prediction
	Parameter
	--------
		list_features : list()
			list of features
	Returns
	-------
		Prediction : str
			Outcome from model
	"""
	model = load_model()
	if model:
		print(data_input)
		result = model.predict(data_input)
		return outcome_display[num_outcome[result[0]]]
	else:
		return "Error in loading model"
	