from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("boston.csv")

# Separate features and target variable
X = data.drop("MV", axis=1)
y = data["MV"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model using pickle
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the trained model from the pickle file
loaded_model = pickle.load(open(filename, 'rb'))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get input values from the form
        CRIM = float(request.form["CRIM"])
        ZN = float(request.form["ZN"])
        INDUS = float(request.form["INDUS"])
        CHAS = int(request.form["CHAS"])
        NOX = float(request.form["NOX"])
        RM = float(request.form["RM"])
        AGE = float(request.form["AGE"])
        DIS = float(request.form["DIS"])
        RAD = int(request.form["RAD"])
        TAX = float(request.form["TAX"])
        PT = float(request.form["PT"])
        B = float(request.form["B"])
        LSTAT = float(request.form["LSTAT"])

        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            "CRIM": [CRIM],
            "ZN": [ZN],
            "INDUS": [INDUS],
            "CHAS": [CHAS],
            "NOX": [NOX],
            "RM": [RM],
            "AGE": [AGE],
            "DIS": [DIS],
            "RAD": [RAD],
            "TAX": [TAX],
            "PT": [PT],
            "B": [B],
            "LSTAT": [LSTAT]
        })

        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_data)[0]

        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)