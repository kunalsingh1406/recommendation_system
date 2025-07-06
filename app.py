from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# data set reading using pandas
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# database configuration---------------------------------------
app.secret_key = "kunal3305"
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:rootpassword@localhost:3306/ecom'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# content based recommendation function 
def content_based_recommendation(df, item_name, top_n):
    #check if the item name exist in the training data
    if item_name not in df['Name'].values:
        print(f"item '{item_name}' not found in training data")
        return pd.DataFrame()

    # create a TF-IDF vectorizer for item description
    tfidf_vec = TfidfVectorizer(stop_words = 'english')
    # creating vectors for different documents
    tfifd_matrix_content = tfidf_vec.fit_transform(df['Tags'])
    # using the vectors and cosine_similarity generating similarity score of one documnet with all other documents
    cosine_similarity_content = cosine_similarity(tfifd_matrix_content, tfifd_matrix_content)

    # finding the idex of the item searched
    item_index = df[df['Name']==item_name].index[0]
    # making a list of similar items of the searched item with the help of cosine_similarity list
    similar_items = list(enumerate(cosine_similarity_content[item_index]))
    # sorting in descending order on the basis of similarity score
    sorted_similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    # choosing the top n highest similarity score products for recommendation
    top_similar_items = sorted_similar_items[0:1+top_n]
    recomended_item_indices = [x[0] for x in top_similar_items]
    recomended_item_details = df.iloc[recomended_item_indices][['Name', 'Brand', 'Rating', 'ReviewCount']]
    return recomended_item_details


#routes
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/main")
def main():
    return render_template('main.html', content_based_rec=pd.DataFrame(), random_price="₹0")


@app.route("/index")
def indexredirect():
    return render_template('index.html')

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()
        return render_template('index.html',signup_message='User signed up successfully!')

@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        new_signup = Signin(username=username,password=password)
        db.session.add(new_signup)
        db.session.commit()
        return render_template('index.html',signup_message='User signed in successfully!')


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr_raw = request.form.get('nbr')

        # Add safe fallback
        try:
            nbr = int(nbr_raw)
        except (ValueError, TypeError):
            return render_template('main.html', message="Please enter a valid number of products.", content_based_rec=None)


        content_based_rec = content_based_recommendation(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)
        else:
            return render_template('main.html', content_based_rec=content_based_rec)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
