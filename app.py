from flask import Flask, request, render_template

from model import predict_class

app = Flask(__name__)


@app.route('/')
def nlp_route():
    return render_template('movieInput.html', size=0)


@app.route('/', methods=['POST'])
def cosine_model():
    reviews = request.form.get('movie')
    review_class = predict_class(str(reviews))
    if review_class == 1:
        return render_template('movieInput.html', review_class="POSITIVE", size=1)
    return render_template('movieInput.html', review_class="NEGATIVE", size=1)


if __name__ == '__main__':
    app.run(port='5000')
