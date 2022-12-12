import streamlit as st

#### Import packages
import streamlit as st
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

#### Add header to describe app
st.header("LinkedIn User App")
st.image('https://kinsta.com/wp-content/uploads/2018/09/linkedin-statistics.png')
st.subheader("Please answer the following questions:")

#Gender
gender = st.radio('Gender', ['Male', 'Female', 'Other'])
if gender == 'Female':
    gender = 1
else:
    gender = 0

#Age
age = st.slider('How old are you?', 0, 97)
st.write("Age:", age)

#Marital 
married = st.radio('Current marital status', ['Married', 'Living with a partner', 'Divorced', 'Seperated', 'Widowed', 'Never been married'])
if married == 'Married':
    married = 1
else:
    married = 0

#Parent
parent = st.radio('Are you a parent?', ['Yes', 'No'])
if parent == "Yes":
    parent = 1
else:
    parent = 0

#Education Level
education = st.selectbox('What is your current education level?', ['Less than high school', 'High school incomplete', 'High school graduate', 'Some college, no degree', 'Two-year associate degree', 'Four-year college or university degree', 'Some postgraduate or professional schooling'])
if education == "Less than high school":
    education = 1
elif education == "High school incomplete":
    education = 2
elif education == "High school graduate":
    education = 3
elif education == "Some college, no degree":
    education = 4
elif education == "Two-year associate degree":
    education = 5
elif education == "Four-year college or university degree":
    education = 6
elif education == "Some postgraduate or professional schooling":
    education = 7
else:
    education = 8
#Income
income = st.selectbox('What is your Income?', ['Less than $10,000', '10 to under $20,000', '20 to under $30,000', '30 to under $40,000', '40 to under $50,000', '50 to under $75,000', '75 to under $100,000', '100 to under $150,000', '$150,000 or more'])
if income =="Less than $10,000":
    income = 1
elif income == "10 to under $20,000":
    income = 2
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
    income = 4
elif income == "40 to under $50,000":
    income = 5
elif income == "50 to under $75,000":
    income = 6
elif income == "75 to under $100,000":
    income = 7
elif income == "100 to under $150,000":
    income = 8
else:
    income = 9

s = pd.read_csv("/Users/sydneypeirce/Desktop/Python Programming 2 Fall 2022/Final Project/social_media_usage.csv")

ss = pd.DataFrame({
    "sm_li":np.where(s["sample"] >= 8, np.nan,
                          np.where(s["sample"] == 1, 1, 0)),
     "income":np.where(s["income"] >= 98, np.nan,
                      s["income"]),
    "education":np.where(s["educ2"] >= 98, np.nan,
                         s["educ2"]),
    "parent":np.where(s["par"] >= 8, np.nan,
                          np.where(s["par"] == 1, 1, 0)),
    "married":np.where(s["marital"] >= 8, np.nan,
                          np.where(s["marital"] == 1, 1, 0)),
    "female":np.where(s["gender"] >= 98, np.nan,
                          np.where(s["gender"] == 2, 1, 0)),
    "age":np.where(s["age"] >= 98, np.nan,
                   s["age"]),})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987)

# Initialize algorithm 
lr = LogisticRegression(class_weight="balanced")
# Fit algorithm to training data
lr.fit(X_train, y_train)

# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

print(classification_report(y_test, y_pred))

person = [age, gender, married, parent, income, education]

# Predict class, given input features
predicted_class = lr.predict([person])

if predicted_class == 1:
    predicted_class_label = "LinkedIn User"
else:
    predicted_class_label = "Not a LinkedIn User"

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

# Print predicted class and probability
if st.button('Are you a LinkedIn user?'):
    st.write(f"This person is a: {predicted_class_label}") # 0=not a LinkedIn user, 1=LinkedIn user
    st.write(f"Probability that this person is a LinkedIn User: {probs[0][1].round(2)}")
else:
    st.write(" ")

