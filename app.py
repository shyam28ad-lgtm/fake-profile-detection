from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        followers = int(request.form["followers"])
        following = int(request.form["following"])
        posts = int(request.form["posts"])
        bio = int(request.form["bio"])
        pic = int(request.form["pic"])
        verified = int(request.form["verified"])

        data = np.array([[followers, following, posts, bio, pic, verified]])

        result = model.predict(data)

        if result[0] == 1:
            output = "⚠️ Fake Profile"
        else:
            output = "✅ Real Profile"

        return render_template("index.html", prediction_text=output)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)