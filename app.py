from flask import Flask, request, render_template, redirect, url_for, session
import sqlite3
import hashlib
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from keras.models import load_model


app = Flask(__name__)

app.secret_key = "super secret key"
app.config['SESSION_TYPE'] = 'filesystem'


# Create a BlobServiceClient using the default Azure credential
# credential = DefaultAzureCredential()
# blob_service_client = BlobServiceClient(account_url="mypsymodel.blob.core.windows.net", credential=credential)

# # Get a reference to the container where the model is stored
# container_client = blob_service_client.get_container_client("newcontainer")

# # Get a reference to the blob that contains the model
# blob_client = container_client.get_blob_client("notes.h5")

# # Download the model to a local file
# with open("notes.h5", "wb") as f:
#     f.write(blob_client.download_blob().readall())

# Load the model from the downloaded file
model = load_model("notes.h5")

# Connect to the database
conn = sqlite3.connect("notes.db", check_same_thread=False)
cursor = conn.cursor()

# Create the tables to store notes and users if they don't already exist
cursor.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, note TEXT, username TEXT)")
cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, notes TEXT)")
conn.commit()

max_words = 5000
max_len= 60

def tok_and_pad(text):
    '''
    This function tokenize the input text into sequnences of intergers and then
    pad each sequence to the same length
    '''
    # tokenisation du texte
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(text)
    # Transformation du texte en séquence d'entier
    X = tokenizer.texts_to_sequences(text)
    # Padding pour mettre chaque sequence à la même longueur
    X = pad_sequences(X, padding='post', maxlen=max_len)
    # affichage
    return X, tokenizer

corpus = pd.read_csv('corpus.csv')

X, tokenizer = tok_and_pad(corpus['clean_text'])
print(X,tokenizer)

def prediction_sentiment(text):
    #Fonction pour predire la categorie de sentiment
    print(text)
    sentiment_categories = ['Negatif', 'Neutre', 'Positif']
    
    max_len=60
    
    # Transforme le texte en une séquence d'entiers
    xt = tokenizer.texts_to_sequences(text)
    print(xt)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    print(xt)
    # Faire la prédiction en utilisant le modèle chargé
    yt = model.predict(xt).argmax(axis=1)


    return sentiment_categories[yt[0]]




# Home page
@app.route("/")
def index():
    if "username" in session:
        return render_template("index.html", logged_in=True)
    else:
        return render_template("index.html", logged_in=False)

# View notes page
@app.route("/view")
def view():
    if "username" not in session:
        return redirect(url_for("index"))
    else:
        # Select all notes belonging to the logged-in user from the database
        cursor.execute("SELECT * FROM notes WHERE username=?", (session["username"],))
        notes = cursor.fetchall()

        # get the note text
        note_text = [note[1] for note in notes]
        print(note_text)
        # Classify the notes as positive, neutral or negative using the function
        labels = [prediction_sentiment([note]) for note in note_text]
        print(labels)
        notes_tuples = [(note, label) for note, label in zip(note_text, labels)]


        # Count the number of positive, neutral and negative notes
        neg_count = sum(1 for label in labels if label == "Negatif")
        neu_count = sum(1 for label in labels if label == "Neutre")
        pos_count = sum(1 for label in labels if label == "Positif")
        
        # Generate a pie chart of the positive, neutral and negative notes
        plt.ion()
        fig, ax = plt.subplots()
        ax.pie([neg_count, neu_count, pos_count], labels=["Negatif", "Neutre", "Positif"], autopct="%1.1f%%")
        plt.title("Notes by Sentiment")
        plt.show()
        # Save the pie chart to a buffer and encode it as a base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()


        return render_template("view.html", image=image_base64, notes=notes_tuples)

# Add note page
@app.route("/add", methods=["GET", "POST"])
def add():
    if "username" not in session:
        return redirect(url_for("index"))
    elif request.method == "POST":
        # Get the note from the form
        note = request.form["note"]

        # Insert the note into the database
        cursor.execute("INSERT INTO notes (username, note) VALUES (?, ?)", (session["username"], note))
        conn.commit()

        # Redirect to the view page
        return redirect(url_for("view"))
    else:
        return render_template("add.html")

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Get the username and password from the form
        username = request.form["username"]
        password = request.form["password"]

        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Check the username and hashed password against the database
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
        user = cursor.fetchone()

        if user is not None:
            # Set the login status in the session
            session["username"] = username
            return redirect(url_for("index"))
        else:
            # Display an error message
            return "Invalid username or password"
    else:
        return render_template("login.html")

# Logout route
@app.route("/logout")
def logout():
    # Clear the login status from the session
    session.clear()
    return redirect(url_for("index"))

# Create user route
@app.route("/create", methods=["GET", "POST"])
def create():
    if request.method == "POST":
        # Get the username and password from the form
        username = request.form["username"]
        password = request.form["password"]

        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Insert the new user into the database
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()

        # Redirect to the login page
        return redirect(url_for("login"))
    else:
        return render_template("create.html")


if __name__ == '__main__':
    app.secret_key = "super secret key"
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=8000, debug=True)
