from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from prediction import predict
from flask import jsonify

import pandas as pd 
import numpy as np 


app = Flask(__name__)
app.secret_key = "super secret key"


@app.route("/")
def home():
    return render_template('index.html')


def check_input(dataframe):
	"""
	This function is used to check whether 
	input is integer or not
	Parameter
	----------
		dataframe : dataframe()
	Returns
	---------
		Returns True if all values is int else False 
	"""
	length_numerical_cols = len(list(dataframe.select_dtypes(
		exclude = 'object').columns))
	length_columns = dataframe.shape[1]

	if length_numerical_cols == length_columns:
		return True
	else:
		return False


@app.route('/recognisediabetic', methods=['GET', 'POST'])
def recognise_diesease():
	"""
	This function helps to predict whether loan should
	be given or not for an applicant
	Parameter
	---------
		Gender
		Married
		Dependents
		Education
		Self_Employed
		Property_area
		Credit_History
		Total_Income
		LoanAmount
		Loan_Amount_Term
	Returns 
	--------
		Prediction from model
	"""
	Gender = int(request.form.get("Gender"))
	Married = int(request.form.get("Married"))
	Dependents = int(request.form.get("Dependents"))
	Education = int(request.form.get("Education"))
	Self_Employed = int(request.form.get("Self_Employed"))
	Property_Area = int(request.form.get("Property_Area"))

	Credit_History = float(request.form.get("Credit_History"))
	Total_Income = float(request.form.get("Total_Income"))
	LoanAmount = float(request.form.get("LoanAmount"))
	Loan_Amount_Term = float(request.form.get("Loan_Amount_Term"))

	dict_test_data = {"Gender" : Gender,
					  "Married" : Married,
					  "Dependents" : Dependents,
    				  "Education" : Education,
                      "Self_Employed" : Self_Employed,
                      "LoanAmount" : LoanAmount,
                      "Total_income" : Total_Income,
                      "Loan_Amount_Term" : Loan_Amount_Term,
                      "Credit_History" : Credit_History,
                      "Property_Area" :Property_Area,
					}

	data_input = pd.DataFrame(dict_test_data, index = [0])
	data_input['LoanAmount_log'] = np.log(data_input['LoanAmount'])
	data_input.drop(['LoanAmount'], axis = 1, inplace = True)


	if check_input(data_input):
		output = predict(data_input)
		return render_template('index.html', 
			prediction = str(output))
	else:
		return render_template('index.html', 
			prediction = str("Input should be integer"))


if __name__ == "__main__":
	app.run(host='127.0.0.1',port=5000,debug=True,threaded=True)
