from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the models and datasets
with open('government_scheme_model.pkl', 'rb') as f:
    text_model, df = pickle.load(f)

with open('model.pkl', 'rb') as f:
    eligibility_model = pickle.load(f)

with open('gender_encoder.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)

with open('state_encoder.pkl', 'rb') as f:
    state_encoder = pickle.load(f)

with open('eligibility_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

clf = model_data['model']
genre_clf = model_data['genre_clf']
le_gender = model_data['le_gender']
le_state = model_data['le_state']
df = model_data['df']

# Function to determine genre based on scheme name
def determine_genre(scheme_name):
    genres = {
        'Healthcare': ['health', 'medical', 'hospital', 'ayushman'],
        'Education': ['education', 'school', 'learning', 'beti bachao', 'skill'],
        'Employment': ['job', 'employment', 'work', 'mgnrega', 'kaushal'],
        'Social Security': ['pension', 'assistance', 'social', 'welfare'],
        'Housing': ['housing', 'home', 'shelter', 'awas'],
        'Digital Empowerment': ['digital', 'technology', 'internet', 'computer'],
        'Financial Inclusion': ['bank', 'finance', 'loan', 'jan dhan', 'mudra'],
        'Women Empowerment': ['women', 'girl', 'matru', 'mahila'],
        'Rural Development': ['rural', 'village', 'gram', 'panchayat'],
        'Urban Development': ['urban', 'city', 'municipal', 'smart city'],
        'Agriculture': ['farm', 'crop', 'agriculture', 'kisan'],
        'Environment': ['environment', 'climate', 'pollution', 'green'],
        'Entrepreneurship': ['entrepreneur', 'startup', 'business', 'stand up india'],
        'Sanitation': ['sanitation', 'toilet', 'hygiene', 'swachh']
    }
    scheme_name = scheme_name.lower()
    for genre, keywords in genres.items():
        if any(keyword in scheme_name for keyword in keywords):
            return genre
    return "General Welfare"

# Function to generate description based on scheme name and genre
def generate_description(scheme_name, genre):
    words = re.findall(r'\w+', scheme_name.lower())
    scheme_type = "flagship" if 'pradhan' in words and 'mantri' in words else "government"
    
    templates = [
        f"A {scheme_type} scheme in the {genre} sector, aimed at improving the lives of citizens through targeted interventions.",
        f"This {genre} initiative focuses on enhancing the welfare of the population through various {scheme_type} measures.",
        f"A comprehensive {scheme_type} program designed to address key issues in the {genre} domain and promote overall development.",
        f"An innovative approach to tackling challenges in the {genre} sector, this {scheme_type} scheme aims to bring about positive change.",
        f"Targeting the {genre} aspect of societal development, this {scheme_type} initiative strives to create a meaningful impact."
    ]
    
    return np.random.choice(templates)

# Function to predict genre using the trained classifier
def predict_genre(scheme_name):
    return genre_clf.predict([scheme_name])[0]

# Function to get scheme information based on scheme name
def get_scheme_info(scheme_name):
    predicted_scheme = text_model.predict([scheme_name])[0]
    scheme = df[df['Scheme Name'] == predicted_scheme].iloc[0]
    
    return {
        'Scheme Name': scheme['Scheme Name'],
        'Genre': scheme['Genre'],
        'Description': scheme['Description'],
        'Eligibility Criteria': f"Applicable Age: {scheme['Applicable Age']}, Gender: {scheme['Gender']}, Income Range: {scheme['Income Range']}"
    }

# Function to check eligibility based on criteria
def check_eligibility(age, gender, state, income):
    eligible_schemes = []
    for _, scheme in df[df['State'] == state].iterrows():
        age_range = scheme['Applicable Age'].split('-')
        
        if len(age_range) == 2:
            min_age, max_age = map(int, age_range)
            if not (min_age <= age <= max_age):
                continue
        elif scheme['Applicable Age'] != 'All Ages' and int(scheme['Applicable Age']) != age:
            continue

        if scheme['Gender'] != 'Both' and scheme['Gender'] != gender:
            continue

        if scheme['State'] != state:
            continue

        if scheme['Income Range'] != 'No Income Limit':
            max_income = int(scheme['Income Range'].split()[2].replace(',', '')) * 100000
            if income > max_income:
                continue

        eligible_schemes.append(scheme['Scheme Name'])

    return eligible_schemes

# Function to get machine learning recommendations
def get_ml_recommendations(age, gender, state, income):
    try:
        gender_encoded = le_gender.transform([gender])[0]
    except ValueError:
        gender_encoded = -1  # Use a default value for unseen labels

    try:
        state_encoded = le_state.transform([state])[0]
    except ValueError:
        state_encoded = -1  # Use a default value for unseen labels

    income_encoded = income // 100000  # Convert to lakhs

    input_data = [[age, gender_encoded, state_encoded, income_encoded]]
    probabilities = clf.predict_proba(input_data)[0]
    
    # Get top 5 recommendations
    top_indices = probabilities.argsort()[-5:][::-1]
    return [clf.classes_[i] for i in top_indices]

# Home route to render the dashboard
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

# Route to render the form for checking eligibility
@app.route('/check-schemes')
def index():
    return render_template('index.html')

# Route to render index2.html
@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/tracking')
def tracking():
    return render_template('tracking.html')

# Route to check eligibility and get recommendations
@app.route('/predict', methods=['POST'])
def check_eligibility_route():
    age = int(request.form['age'])
    gender = request.form['gender']
    state = request.form['state']
    income = float(request.form['income'])

    # Get eligible schemes
    eligible_schemes = check_eligibility(age, gender, state, income)

    # Get machine learning recommendations for schemes
    recommendations = get_ml_recommendations(age, gender, state, income)

    # Prepare the response data
    response_data = {
        'eligible_schemes': eligible_schemes,
        'recommendations': recommendations,
        'count': len(eligible_schemes)
    }

    return jsonify(response_data)

# Route to get scheme information based on scheme name
@app.route('/get_scheme_info', methods=['POST'])
def get_scheme_info_route():
    scheme_name = request.form['scheme_name']
    scheme_info = get_scheme_info(scheme_name)
    return jsonify(scheme_info)





if __name__ == '__main__':
    app.run(debug=True)
