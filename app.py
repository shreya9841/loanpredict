from flask import Flask, request, render_template, redirect, url_for, session
import pickle
import numpy as np
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from model import TreeNode, DecisionTree, RandomForest  # your custom model

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Load ML model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Encoding maps
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"Yes": 1, "No": 0}
property_area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

# Home
@app.route("/")
def index():
    return render_template("index.html")

# Sign up
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Username already exists.")
        finally:
            conn.close()
    return render_template("signup.html")

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['user'] = username
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")

# Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

# Loan prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            # Get form data
            gender = request.form.get("Gender")
            married = request.form.get("Married")
            dependents = request.form.get("Dependents")
            education = request.form.get("Education")
            self_employed = request.form.get("Self_Employed")
            applicant_income = float(request.form.get("ApplicantIncome"))
            coapplicant_income = float(request.form.get("CoapplicantIncome"))
            loan_amount = float(request.form.get("LoanAmount"))
            loan_amount_term = float(request.form.get("Loan_Amount_Term"))
            credit_history = float(request.form.get("Credit_History"))
            property_area = request.form.get("Property_Area")

            # Encode inputs
            gender_enc = gender_map.get(gender, 0)
            married_enc = married_map.get(married, 0)
            dependents_enc = dependents_map.get(dependents, 0)
            education_enc = education_map.get(education, 0)
            self_employed_enc = self_employed_map.get(self_employed, 0)
            property_area_enc = property_area_map.get(property_area, 0)

            input_dict = {
                'Gender': gender_enc,
                'Married': married_enc,
                'Dependents': dependents_enc,
                'Education': education_enc,
                'Self_Employed': self_employed_enc,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': property_area_enc
            }

            input_data = np.array([[input_dict[feature] for feature in feature_names]])

            prediction = model.predict(input_data)[0]
            pred_label = "Y" if prediction == 1 else "N"

            return render_template("form.html", prediction=pred_label,
                                   input_data=input_data.tolist(),
                                   raw_prediction=prediction)

        except Exception as e:
            return render_template("form.html", error=str(e))

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
