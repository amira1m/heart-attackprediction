What it does
Takes patient health features (such as age, cholesterol, blood pressure, etc.) and predicts whether the patient is at risk of a heart attack. Multiple ML models are trained, compared, and saved for reuse.
Models Used

Logistic Regression
Random Forest
Support Vector Machine (SVM — RBF kernel)
Polynomial Regression
Multiple Linear Regression

How it works

Preprocessing — Cleans and prepares the dataset, handles missing values, and selects relevant features (preprocessing.py, feature_selection.py)
Model Training — Trains and evaluates multiple regression and classification models (regg.py)
Model Saving — Saves trained models as .pkl files for reuse without retraining
Evaluation — Compares models using accuracy and R² score

Tech Stack

Python
Scikit-learn
Pandas
NumPy
