from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        followers = int(request.form.get("followers", 0))
        following = int(request.form.get("following", 0))
        posts = int(request.form.get("posts", 0))
        bio = int(request.form.get("bio", 0))
        pic = int(request.form.get("pic", 0))
        verified = int(request.form.get("verified", 0))

        features = [[followers, following, posts, bio, pic, verified]]
        prediction = model.predict(features)

        result = "Fake Profile" if prediction[0] == 1 else "Real Profile"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {e}"
