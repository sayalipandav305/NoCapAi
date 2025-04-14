
from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from utils.image_analysis import analyze_image
from utils.video_analysis import analyze_video

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                result = analyze_image(filepath)
            elif filename.lower().endswith(('mp4', 'mov')):
                result = analyze_video(filepath)
                
            os.remove(filepath)
        else:
            result = "Invalid file format."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
