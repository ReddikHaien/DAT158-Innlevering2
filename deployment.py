from fastai.vision.all import *
from fastai.data.external import *
from flask import *
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Imag data
    data = urllib.request.urlopen(request.form["inputImage"])
    bytes = data.file.read()
    image = Image.open(io.BytesIO(bytes))


    #scaling and moving
    offsetx = int(request.form["offsetx"])
    offsety = int(request.form["offsety"])
    scale = int(request.form["scale"])
    
    
    width = image.width
    height = image.height

    width *= scale / 100.0
    height *= scale / 100.0

    scaledImage = image.resize((int(width),int(height)))

    croppedImage = scaledImage.crop((offsetx,offsety, offsetx + 224, offsety + 224))

    arrs = np.array(croppedImage)

    print(arrs.shape)

    pilimage = PILImage.create(arrs)

    model = load_learner("modell.pkl")
    
    prediction = model.predict(pilimage)

    species = prediction[0]
    index = prediction[1].item()
    score = prediction[2][index]

    return f"{{\"species\":\"{species}\", \"score\":{score}}}"