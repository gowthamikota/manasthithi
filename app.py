from flask import Flask, render_template, request
import numpy as np
import pickle
import requests  # Import the requests library to make API calls
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the model, scaler, and label encoders
with open('student_mental_health_model (1).pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

label_encoder_sleep = LabelEncoder()
label_encoder_diet = LabelEncoder()
label_encoder_gender = LabelEncoder()

label_encoder_sleep.fit(['5-6 hours', '7-8 hours', 'More than 8 hours'])
label_encoder_diet.fit(['Unhealthy', 'Moderate', 'Healthy'])
label_encoder_gender.fit(['Male', 'Female'])

# Fetch blogs from NewsAPI filtered by the 'mental health' topic
def fetch_blogs():
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': 'mental health',  # Search query for mental health
        'apiKey': '135f086b820045d383abbad378206e98',
        'language': 'en',  # English articles
        'pageSize': 5  # Limit to the latest 5 articles
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['articles']  # Return the list of mental health-related articles
    else:
        return []  # Return an empty list if the API request fails

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'Gender': request.form['Gender'],
            'Age': int(request.form['Age']),
            'Academic Pressure': int(request.form['Academic Pressure']),
            'Study Satisfaction': int(request.form['Study Satisfaction']),
            'Sleep Duration': request.form['Sleep Duration'],
            'Dietary Habits': request.form['Dietary Habits'],
            'Have you ever had suicidal thoughts?': request.form['Suicidal Thoughts'],
            'Study Hours': int(request.form['Study Hours']),
            'Financial Stress': int(request.form['Financial Stress'])
        }

        # Preprocess user input
        gender = label_encoder_gender.transform([user_input['Gender']])[0]
        sleep = label_encoder_sleep.transform([user_input['Sleep Duration']])[0]
        diet = label_encoder_diet.transform([user_input['Dietary Habits']])[0]
        studied = 1 if user_input['Have you ever had suicidal thoughts?'] == 'Yes' else 0

        # Combine features into an array
        new_data = np.array([ 
            gender, user_input['Age'], user_input['Academic Pressure'], user_input['Study Satisfaction'], 
            sleep, diet, studied, user_input['Study Hours'], user_input['Financial Stress']
        ]).reshape(1, -1)

        # Standardize numerical features
        new_data_scaled = scaler.transform(new_data)

        # Make the prediction
        prediction = model.predict(new_data_scaled)
        risk_proba = model.predict_proba(new_data_scaled)[0][1]

        # Display the prediction result
        if prediction[0] == 1:
            result = f"The user is at risk of depression. Risk Probability: {risk_proba * 100:.2f}%"
        else:
            result = f"The user is not at risk of depression. Risk Probability: {risk_proba * 100:.2f}%"

        return render_template('result.html', result=result)

# Blog route to display list of blogs
@app.route('/blog')
def blog():
    blogs = fetch_blogs()  # Fetch the list of mental health-related blogs from the NewsAPI
    if not blogs:  # If no blogs are fetched, show a message
        message = "No blogs available at the moment. Please try again later."
    else:
        message = None
    return render_template('blog.html', blogs=blogs, message=message)

# Blog detail route to display a specific blog post
@app.route('/blog/<path:blog_url>')  # Using 'path' to capture the entire URL as a string
def blog_detail(blog_url):
    blogs = fetch_blogs()  # Fetch all blogs again (you could optimize this by caching the list)
    blog_post = next((blog for blog in blogs if blog['url'] == blog_url), None)
    if blog_post is None:
        return "Blog post not found", 404  # Return a 404 if the blog is not found
    return render_template('blog_detail.html', blog=blog_post)

# Other routes
@app.route('/test')
def assessment():
    return render_template('assessment.html')




if __name__ == "__main__":
    app.run(debug=True)
