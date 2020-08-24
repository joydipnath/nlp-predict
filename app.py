from flask import Flask, jsonify, request, render_template
from flask_restful import Api, Resource, reqparse
from flask_jwt import JWT, jwt_required, current_identity
from flask_cors import CORS, cross_origin
from secure_check import authenticate, identity
from flask_mongoengine import MongoEngine
from sentiment_analysis import SentimentAnalysis

app = Flask(__name__)
app.config['SECRET_KEY'] = '123456789'
api = Api(app)
jwt = JWT(app, authenticate, identity)


class Add(Resource):
    """docstring for Add."""

    # def __init__(self, arg):
    #     super(Add, self).__init__(self, arg)
    #     self.arg = arg

    def post(self, name):
        posteddata = request.get_json(force=True)
        x = int(posteddata['x'])
        y = int(posteddata['y'])
        ret = x + y
        returnmap = {
            'msg': "",
            'val': ret,
            'status_code': 200
        }
        return jsonify(returnmap)

    def get(self, name):
        return jsonify({'hi': name})


puppies = []


class Puppies(Resource):

    def post(self, name):
        puppies.append({"name": name})
        return puppies

    @jwt_required()
    def get(self, name):
        for pup in puppies:
            if pup['name'] == name:
                return pup
        return {'name': None}

    def delete(self, name):
        for index,value in enumerate(puppies):
            if value['name'] == name:
                deleted_pup = puppies.pop(index)
                return {'note': 'deleted successfully'}


api.add_resource(Add, '/add/<string:name>')
api.add_resource(Puppies, '/puppies/<string:name>')


@app.route('/', methods=["GET"])
def index():
    name='Joydip'
    # print(app.config['DB_NAME'])
    return render_template('index.html', name=name)


@app.route('/puppy/<name>', methods=["GET"])
def puppy_profile(name):
    return "this is a puppy {} profile".format(name)


@app.route('/predict/sentiment', methods=["POST"])
def sentiment():
    senti = SentimentAnalysis()
    input_msg = request.form.get('sentiment_text')
    prediction = senti.sentiment(input_msg)
    if prediction > 0.80:
        message = 'It is a positive message, having a score of {}'.format(prediction)
    else:
        message = 'It is a negative message, having a score of {}'.format(prediction)
    return render_template('result.html', prediction=message, input=input_msg)


if __name__ == "__main__":
    app.run(debug=True)
