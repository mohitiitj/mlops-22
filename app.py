from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model = load('./svm_gamma=0.001_C=0.5.joblib')

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow    
#     - query parameter
#     - get call / methods
#     - post call / methods ** 

@app.route("/sum", methods=['POST'])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y 
    return {'sum':z}



@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    print("done loading")
    predicted = model.predict([image])
    return {"y_predicted":int(predicted[0])}

@app.route("/checkSame", methods=['POST'])
def checkSame():
    image1 = request.json['image1']
    image2 = request.json['image2']
    predicted1 = model.predict([image1])
    predicted2 = model.predict([image2])
    if predicted1 == predicted2:
        return "Images are same"
    return "Images are not same"
