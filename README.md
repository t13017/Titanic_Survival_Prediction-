# Titanic_Survival_Prediction-
# Overview

The Titanic Survival Prediction Project is a machine learning-based project that aims to predict the survival of passengers onboard the famous Titanic ship. The project utilizes three different machine learning models, namely XGBoost, Random Forest, and Logistic Regression, to predict whether a passenger survived the tragic event or not.

# Project Structure

The project consists of the following files and directories:

•	FE_titanic.csv: The dataset file containing preprocessed features and target variable (Survived).

•	Titanic.ipynb: A Jupyter Notebook containing the data analysis, model training, and evaluation steps.

•	app.ipynb: The Flask API script for deploying the trained models and making predictions.

•	logistic_regression_model.joblib: The serialized Logistic Regression model saved using joblib.

•	xgboost_model.joblib: The serialized XGBoost model saved using joblib.

•	random_forest_model.joblib: The serialized Random Forest model saved using joblib.

•	templates/index.html: The HTML template for the web interface to interact with the Flask API.

# Data Analysis and Model Training

The Jupyter Notebook Titanic_Survival_Prediction.ipynb contains detailed data analysis, data preprocessing, feature engineering, and model training steps. The analysis explores the dataset to understand its structure and relationships between variables. Features like Pclass, Age, Fare, Parch, SibSp, SexIndex, and EmbarkedIndex are used as inputs, while survived is used as the target variable.

Three different machine learning models, XGBoost, Random Forest, and Logistic Regression, are trained using the processed data. Hyperparameter tuning is performed using GridSearchCV to find the best set of hyperparameters for each model, optimizing their performance.
The trained models are then serialized and saved using joblib to be used in the Flask API for predictions.

# Flask API and Web Interface
The Flask API serves as the backend for making predictions using the trained models. It uses the joblib library to load the serialized models, and it provides an HTML-based web interface to interact with the models. The web interface allows users to input information about a passenger, such as Pclass, Age, Fare, Parch, SibSp, SexIndex, and EmbarkedIndex. The prediction result is then displayed on the web interface. We have provided an interface for all three models separately.

# Instructions to Run the Project

To run the Titanic Survival Prediction Project:

•	Ensure you have Python and the required libraries (pandas, scikit-learn, xgboost, joblib, flask) installed on your system.

•	Clone or download the project repository.

•	Open a terminal or command prompt and navigate to the project directory.

•	Run the colab Notebook Titanic.ipynb to perform data analysis and model training.

•	After running the notebook, three model files (logistic_regression_model.joblib, xgboost_model.joblib, and random_forest_model.joblib) will be created in the 
same directory.

•	Run the Flask API by executing the command python app.py in the terminal.

•	The API will start running locally, and you will see a URL for the web interface (usually http://127.0.0.1:5000/).

•	Open your web browser and navigate to the URL provided by the Flask API to access the web interface.

•	Enter the details of a passenger, and the model will predict the survival status of the passenger.

# Conclusion

The Titanic Survival Prediction Project showcases the use of different machine learning algorithms to predict the survival of passengers on the Titanic. The combination of data analysis, preprocessing, and model training allows for accurate predictions using the Flask API and web interface. Users can now easily interact with the trained models to gain insights into the passenger survival based on their input information.
