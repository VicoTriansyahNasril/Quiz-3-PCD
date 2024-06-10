import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response, url_for, send_file
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
from dotenv import set_key, load_dotenv, dotenv_values
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
load_dotenv()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER_IMAGE = 'static/img'
UPLOAD_FOLDER_VIDEO = 'static/video'
ALLOWED_EXTENSIONS_IMAGE = {'jpg', 'jpeg', 'png', 'gif'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER_IMAGE'] = UPLOAD_FOLDER_IMAGE
app.config['UPLOAD_FOLDER_VIDEO'] = UPLOAD_FOLDER_VIDEO


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")


@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/upload", methods=["POST"])
@nocache
def upload():
    if 'file' not in request.files:
        return 'No file part'

    files = request.files.getlist("file")
    if len(files) == 0:
        return 'No selected file'

    file = files[0]
    file_type = file.filename.rsplit('.', 1)[1].lower()

    if file_type in ALLOWED_EXTENSIONS_IMAGE:
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_IMAGE):
            file.save("static/img/img_now.jpg")
            copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
            return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')
    elif file_type in ALLOWED_EXTENSIONS_VIDEO:
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_VIDEO):
            file.save("static/video/video_now.mp4")
            return render_template("uploaded.html", file_path="video/video_now.mp4", file_type='video')

    return 'Invalid file type'


@app.route('/uploads/<file_type>/<filename>')
def uploaded_file(file_type, filename):
    if file_type == 'img':
        return send_file(os.path.join(app.config['UPLOAD_FOLDER_IMAGE'], filename))
    elif file_type == 'video':
        return send_file(os.path.join(app.config['UPLOAD_FOLDER_VIDEO'], filename))


@app.route('/recognize', methods=['POST'])
def recognize():
    chain_codes = image_processing.load_chain_codes_from_env()
    image_path = "static/img/img_now.jpg"
    recognized_emoji = image_processing.recognize_emoji(
        image_path, chain_codes)
    return render_template("uploaded.html", predicted_emoji=recognized_emoji, file_path="img/img_now.jpg", file_type='image')


@app.route("/imgclass", methods=["POST"])
@nocache
def classify():
    image_processing.classify_image("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/detect_objects", methods=["POST"])
@nocache
def detect_objects():
    image_processing.detect_objects("static/img/img_now.jpg", threshold=0.9)
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/video_classify", methods=["POST"])
@nocache
def video_classify():
    video_path = "static/video/video_now.mp4"
    image_processing.classify_video(video_path)
    return render_template("uploaded.html", file_path="video/video_now.mp4", file_type="video")


@app.route("/normal", methods=["POST"])
@nocache
def normal():
    copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    image_processing.grayscale()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/binary", methods=["POST"])
@nocache
def binary():
    image_processing.binary_image(128)
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/erosion", methods=["POST"])
@nocache
def erosion():
    image_processing.erosion()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/dilation", methods=["POST"])
@nocache
def dilation():
    image_processing.dilation()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/opening", methods=["POST"])
@nocache
def opening():
    image_processing.opening()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/closing", methods=["POST"])
@nocache
def closing():
    image_processing.closing()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/count", methods=["POST"])
@nocache
def count():
    num_objects = image_processing.count_shattered_glass()
    return render_template("uploaded.html", num_objects=num_objects, file_path="img/img_now.jpg", file_type='image')


@app.route("/countsquare", methods=["POST"])
@nocache
def countsquare():
    num_objects = image_processing.count_square()
    return render_template("uploaded.html", num_objects=num_objects, file_path="img/img_now.jpg", file_type='image')


@app.route("/countcell", methods=["POST"])
@nocache
def countcell():
    num_objects = image_processing.count_objects()
    return render_template("uploaded.html", num_objects=num_objects, file_path="img/img_now.jpg", file_type='image')


@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    image_processing.zoomin()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    image_processing.zoomout()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/move_left", methods=["POST"])
@nocache
def move_left():
    image_processing.move_left()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/move_right", methods=["POST"])
@nocache
def move_right():
    image_processing.move_right()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/move_up", methods=["POST"])
@nocache
def move_up():
    image_processing.move_up()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/move_down", methods=["POST"])
@nocache
def move_down():
    image_processing.move_down()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/brightness_addition", methods=["POST"])
@nocache
def brightness_addition():
    image_processing.brightness_addition()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/brightness_substraction", methods=["POST"])
@nocache
def brightness_substraction():
    image_processing.brightness_substraction()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/brightness_multiplication", methods=["POST"])
@nocache
def brightness_multiplication():
    image_processing.brightness_multiplication()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/brightness_division", methods=["POST"])
@nocache
def brightness_division():
    image_processing.brightness_division()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/histogram_equalizer", methods=["POST"])
@nocache
def histogram_equalizer():
    image_processing.histogram_equalizer()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/edge_detection", methods=["POST"])
@nocache
def edge_detection():
    image_processing.edge_detection()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/blur", methods=["POST"])
@nocache
def blur():
    image_processing.blur()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/sharpening", methods=["POST"])
@nocache
def sharpening():
    image_processing.sharpening()
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    image_processing.histogram_rgb()
    if image_processing.is_grey_scale("static/img/img_now.jpg"):
        return render_template("histogram.html", file_paths=["img/grey_histogram.jpg"], file_type='image')
    else:
        return render_template("histogram.html", file_paths=["img/red_histogram.jpg", "img/green_histogram.jpg", "img/blue_histogram.jpg"], file_type='image')


@app.route("/thresholding", methods=["POST"])
@nocache
def thresholding():
    lower_thres = int(request.form['lower_thres'])
    upper_thres = int(request.form['upper_thres'])
    image_processing.threshold(lower_thres, upper_thres)
    return render_template("uploaded.html", file_path="img/img_now.jpg", file_type='image')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
