from flask import Flask, request, render_template, redirect, url_for, session
import pickle
import numpy as np
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from model import TreeNode, DecisionTree, RandomForest  # your custom model
from flask_mysqldb import MySQL   # <-- import here


app = Flask(__name__)
app.secret_key = "your_secret_key_here"

from flask_mysqldb import MySQL

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''  
app.config['MYSQL_DB'] = 'loan_system'

mysql = MySQL(app)

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

        cur = mysql.connection.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_pw))
            mysql.connection.commit()
            return redirect(url_for("login"))
        except Exception:
            return render_template("signup.html", error="Username already exists.")
        finally:
            cur.close()
    return render_template("signup.html")



# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[2], password):  # user[2] = password column
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
            input_dict = {
                'Gender': gender_map.get(gender, 0),
                'Married': married_map.get(married, 0),
                'Dependents': dependents_map.get(dependents, 0),
                'Education': education_map.get(education, 0),
                'Self_Employed': self_employed_map.get(self_employed, 0),
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': property_area_map.get(property_area, 0)
            }

            input_data = np.array([[input_dict[feature] for feature in feature_names]])
            prediction = model.predict(input_data)[0]
            pred_label = "Approved ✅" if prediction == 1 else "Rejected ❌"

            return render_template("result.html",
                                   prediction=pred_label,
                                   input_data=input_dict)

        except Exception as e:
            return render_template("form.html", error=str(e))

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
