import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Function to display landing page
def landing_page():
    st.markdown(
        """
        <style>
        .main {
            background-color: black;
            color: white;
        }
        .stApp {
            background-color: black;
        }
        .title {
            color: #ffccff;
            font-size: 3em;
            font-weight: bold;
        }
        .slogan {
            color: #ffccff;
            font-size: 1.5em;
            font-style: italic;
        }
        .description {
            color: #ffffff;
            font-size: 1.2em;
        }
        .option-text {
            font-size: 1.1em;
            color: #ffccff;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #ffccff;
            color: black;
            text-align: center;
            padding: 10px 0;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="title">Banking Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="slogan">Predict and Prevent Customer Churn Effectively</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="description">Welcome to our Customer Churn Prediction tool! This application helps banks predict which customers are likely to leave, allowing for timely interventions to retain valuable clients.</p>', 
        unsafe_allow_html=True
    )

    st.markdown('<p class="option-text">Choose an option:</p>', unsafe_allow_html=True)
    option = st.selectbox('', ('Upload CSV Data', 'Enter Customer Details Manually'))
    return option

# Function to upload CSV data
def upload_csv():
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        return None

# Function for manual input
def manual_input():
    st.sidebar.header('Customer Input Features')
    def user_input_features():
        customer_id = st.sidebar.number_input('Customer ID', min_value=1, value=1)
        age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
        gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
        tenure = st.sidebar.slider('Tenure', min_value=0, max_value=10, value=5)
        balance = st.sidebar.number_input('Balance', min_value=0, value=50000)
        products = st.sidebar.slider('Number of Products', min_value=1, max_value=4, value=2)
        credit_score = st.sidebar.slider('Credit Score', min_value=300, max_value=900, value=600)
        active_member = st.sidebar.selectbox('Active Member', ('Yes', 'No'))
        estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0, value=50000)

        data = {
            'customer_id': customer_id,
            'age': age,
            'gender_Male': 1 if gender == 'Male' else 0,
            'tenure': tenure,
            'balance': balance,
            'products': products,
            'credit_score': credit_score,
            'active_member_Yes': 1 if active_member == 'Yes' else 0,
            'estimated_salary': estimated_salary
        }
        return pd.DataFrame(data, index=[0])
    return user_input_features()

# Function to preprocess data
def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df

# Function to display prediction results
def display_results(input_df, model, scaler):
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_df = scaler.transform(input_df)
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    st.write('**Churn**' if prediction[0] else '**No Churn**')

    st.subheader('Prediction Probability')
    st.write(f"Churn Probability: **{prediction_proba[0][1]:.2f}**")
    st.write(f"No Churn Probability: **{prediction_proba[0][0]:.2f}**")

# Function to show data details
def show_data_details(data):
    if st.checkbox('Show Data Details'):
        st.subheader('Data Details')
        st.write(data.describe())
        st.write(data)

        st.subheader('Customer Data Distribution')
        fig, ax = plt.subplots()
        sns.histplot(data['age'], kde=True, ax=ax, color='purple')
        ax.set_title('Age Distribution')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.histplot(data['balance'], kde=True, ax=ax, color='green')
        ax.set_title('Balance Distribution')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.countplot(x='products', data=data, ax=ax, palette='cool')
        ax.set_title('Number of Products Distribution')
        st.pyplot(fig)

# Load the sample data
sample_data = pd.read_csv('customer_data.csv')

# Preprocess the sample data
sample_data = preprocess_data(sample_data)

# Split the sample data into training and testing sets
X = sample_data.drop('churn', axis=1)
y = sample_data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit App
option = landing_page()

if option == 'Upload CSV Data':
    data = upload_csv()
    if data is not None:
        st.write("Data Uploaded Successfully")
        st.write("Displaying a few rows of the data:")
        st.write(data.head())
        show_data_details(data)
else:
    input_df = manual_input()
    display_results(input_df, model, scaler)

# Footer
st.markdown(
    """
    <div class="footer">
        <p>By Elhaimer Salma</p>
    </div>
    """, 
    unsafe_allow_html=True
)
