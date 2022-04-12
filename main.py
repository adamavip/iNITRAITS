from operator import eq
from flask import Flask, render_template, redirect, url_for, session, request, flash, send_from_directory
import os
from prepare_data import load_test_data
from prediction import infer
import pandas as pd


app = Flask(__name__)

# Configure session key
app.secret_key = "1428245666"

# Configure path to uploaded files
UPLOAD_FOLDER = './static/file/uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define crops and sensors
crops = ["Peanut", "Sorghum", "Millet"]
equipments = ["BENCHTOP", "EVT5", "HLR"]


@app.route("/")
def home():
    return render_template("index.html",crops=crops)

@app.route("/sensor/<crop_name>")
def sensor(crop_name):
    session["cropname"] = crop_name
    return render_template("sensor.html",crop_name = crop_name, equipments=equipments)


@app.route("/sensor/<crop_name>/<equipment>")
def device(crop_name, equipment):
    cropname = session.get("cropname")
    session["equipment"] = equipment
    return render_template("model.html",crop_name=cropname,equipment=equipment)



@app.route("/upload_file", methods=["GET", "POST"])
def upload_file():
    predicted_data=None

    def allowed_file(filename):
        return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    if request.method == 'POST':
    # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath) 
            session["filepath"] = filepath
            return redirect(url_for('uploaded_file',filename=filename))

    return render_template("model.html", crop_name=session.get("cropname"), 
            equipment=session.get("equipment"))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
        

@app.route("/predict")
def predict():
    # read and preprocess data
    filepath = session.get("filepath")
    processed_data = load_test_data(filepath)

    # Make predictions using pretrained model
    predicted_data = infer(processed_data)  
    colnames = predicted_data.columns
    rows = predicted_data.to_numpy()

    return render_template("prediction.html", crop_name=session.get("cropname"), 
            equipment=session.get("equipment"), colnames=colnames, 
            rows = rows)





@app.route("/test_upload")
def test_upload():
    df = pd.read_excel("D:/Code/Python/starch-nirs-model/results/predictions/Millet_DigestibleStarch_val_CNN_HLR_09-Apr-2022_21-50-52_.xlsx")
    
    colnames = df.columns
    rows = df.to_numpy()

    return render_template("prediction.html", crop_name=session.get("cropname"), 
                        equipment=session.get("equipment"), colnames=colnames, 
                        rows = rows)

if __name__ == '__main__':
    app.run(debug=True)

