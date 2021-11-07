from flask import Flask, render_template, request
import os
import classifier as clf

templates_dir = os.path.join(os.curdir, "templates")
static_dir = os.path.join(os.curdir,"static")

app = Flask(__name__, template_folder=templates_dir,static_folder=static_dir)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        oneRecord = clf.prepare_data(request.form)
        result = clf.predict(oneRecord)
        if len(result) > 0:
            prediction = "Diabetic And Frequently visits Hospital (Severe Condition)" if result[0] == 1 else "Not Diabetic And Doesn't Visit Hospital (Healthy or Mild Condition)"
        else:
            prediction = result
        return render_template('predict.html', prediction=prediction)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
