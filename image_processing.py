import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import Counter
import cv2
from dotenv import set_key, load_dotenv, dotenv_values
import torch
import os


from transformers import ViTFeatureExtractor, ViTForImageClassification, DetrImageProcessor, DetrForObjectDetection,  VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from decord import VideoReader, cpu

load_dotenv()

model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')


def classify_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        'google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    confidence_score = torch.softmax(
        logits, dim=-1)[0, predicted_class_idx].item()

    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class, confidence_score


def detect_objects(image_path, threshold=0.9):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    target_sizes = torch.tensor(image.shape[:2]).unsqueeze(0)
    postprocessed_outputs = image_processor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
    scores = postprocessed_outputs[0]['scores'][keep]
    labels = postprocessed_outputs[0]['labels'][keep]

    for bbox, score, label in zip(bboxes_scaled, scores, labels):
        x, y, w, h = bbox.tolist()

        x = max(0, int(x))
        y = max(0, int(y))
        w = min(width, int(w))
        h = min(height, int(h))

        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

        class_name = model.config.id2label[label.item()]
        label_text = f"{class_name}: {score.item():.2f}"

        label_y = max(15, y - 10)

        label_x = x
        label_width = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
        if label_x + label_width > width:
            label_x = width - label_width

        cv2.putText(image, label_text, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite('static/img/img_now.jpg', image)


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    str_idx = end_idx - converted_len
    index = np.linspace(str_idx, end_idx, num=clip_len)
    index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
    return index


def classify_video(video_path):
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name)

    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))

    vr.seek(0)
    clip_len = 16
    frame_sample_rate = 4
    seg_len = len(vr)
    index = sample_frame_indices(clip_len, frame_sample_rate, seg_len)
    buffer = vr.get_batch(index).asnumpy()

    inputs = feature_extractor(list(buffer), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        confidence_score = torch.softmax(
            logits, dim=-1)[0, predicted_class_idx].item()
        predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class, confidence_score


def freeman_chain_code(contour):
    chain_code = []
    start_point = contour[0][0]
    current_point = start_point

    for point in contour[1:]:
        x, y = point[0]
        dx = x - current_point[0]
        dy = y - current_point[1]
        direction = None

        if dx == 1 and dy == 0:
            direction = 0
        elif dx == 1 and dy == -1:
            direction = 1
        elif dx == 0 and dy == -1:
            direction = 2
        elif dx == -1 and dy == -1:
            direction = 3
        elif dx == -1 and dy == 0:
            direction = 4
        elif dx == -1 and dy == 1:
            direction = 5
        elif dx == 0 and dy == 1:
            direction = 6
        elif dx == 1 and dy == 1:
            direction = 7

        if direction is not None:
            chain_code.append(direction)
            current_point = (x, y)

    return chain_code

# Fungsi untuk thinning citra


def thinning(image):
    img_arr = np.array(image)
    size = np.size(img_arr)
    skel = np.zeros(img_arr.shape, np.uint8)
    ret, img = cv2.threshold(img_arr, 127, 255, cv2.THRESH_BINARY_INV)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

# Fungsi untuk memuat kode rantai dari file .env


def load_chain_codes_from_env():
    data = dotenv_values(".env")
    chain_codes = {}

    for key, value in data.items():
        if key.endswith('_chain_code'):
            emoji_name = key.rsplit('_', 1)[0]
            chain_code = list(map(int, value.split(',')))
            chain_codes[emoji_name] = chain_code

    return chain_codes


# Fungsi untuk mengenali emoji dari gambar


def recognize_emoji(image_path, chain_codes):
    # Fungsi untuk mendapatkan chain code dari gambar
    chain_codes_image = get_chain_code_from_image(image_path)
    if chain_codes_image is None:
        return "Unable to recognize emoji. Error reading image."

    # Mencocokkan chain code dari gambar dengan daftar chain codes yang diberikan
    for emoji_name, emoji_chain_code in chain_codes.items():
        if chain_codes_image == emoji_chain_code:
            # Hilangkan kata 'chain' dari nama emoji jika ada
            if emoji_name.endswith('chain'):
                emoji_name = emoji_name[:-5]
            return emoji_name

    # Jika tidak ada kecocokan, kembalikan nama file lengkap dengan ekstensi
    file_name = os.path.basename(image_path)
    return file_name


# Fungsi untuk mendapatkan Freeman Chain Code dari gambar


def get_chain_code_from_image(image_path):
    emoji_image = cv2.imread(image_path, 0)

    if emoji_image is None:
        print("Error membaca gambar")
        return None

    thin_img = thinning(emoji_image)
    contours, _ = cv2.findContours(
        thin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    total_chain_code = []
    for contour in contours:
        chain_code = freeman_chain_code(contour)
        total_chain_code.extend(chain_code)

    return total_chain_code


def grayscale():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    new_arr = ((r.astype(int) + g.astype(int) +
               b.astype(int)) // 3).astype(np.uint8)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def binary_image(threshold_value):
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).copy()
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    _, binary_img = cv2.threshold(
        img_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    new_img = Image.fromarray(binary_img)
    new_img.save("static/img/img_now.jpg")


def count_objects():
    img = cv2.imread('static/img/img_now.jpg', 0)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_objects = len(contours)
    return num_objects


def count_square():
    # Baca citra sebagai citra keabuan (grayscale)
    img_gray = cv2.imread('static/img/img_now.jpg', cv2.IMREAD_GRAYSCALE)

    # Lakukan thresholding untuk mendapatkan citra biner
    # Ubah nilai threshold sesuai kebutuhan
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

    # Temukan kontur objek pada citra biner
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hitung jumlah objek (kontur) yang ditemukan
    num_objects = len(contours)

    return num_objects


def erosion():
    img = Image.open("static/img/img_now.jpg")
    kernel = np.ones((5, 5), np.uint8)
    eroded_img = cv2.erode(np.array(img), kernel, iterations=1)
    new_img = Image.fromarray(eroded_img)
    new_img.save("static/img/img_now.jpg")


def dilation():
    img = Image.open("static/img/img_now.jpg")
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(np.array(img), kernel, iterations=1)
    new_img = Image.fromarray(dilated_img)
    new_img.save("static/img/img_now.jpg")


def opening():
    img = Image.open("static/img/img_now.jpg")
    kernel = np.ones((5, 5), np.uint8)
    opened_img = cv2.morphologyEx(np.array(img), cv2.MORPH_OPEN, kernel)
    new_img = Image.fromarray(opened_img)
    new_img.save("static/img/img_now.jpg")


def closing():
    open_img = Image.open("static/img/img_now.jpg")
    kernel = np.ones((5, 5), np.uint8)
    closed_img = cv2.morphologyEx(np.array(open_img), cv2.MORPH_CLOSE, kernel)
    new_img = Image.fromarray(closed_img)
    new_img.save("static/img/img_now.jpg")


def count_shattered_glass():
    img = Image.open("static/img/img_now.jpg").convert('L')
    img_np = np.array(img)
    img_blur = cv2.GaussianBlur(img_np, (5, 5), 0)
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.4*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    num_shattered_glass = len(np.unique(markers))
    img_color[markers == -1] = [255, 0, 0]
    return num_shattered_glass


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g or g != b:
                return False
    return True


def zoomin():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.array(img)

    # Double the size of the image
    new_size = (img_arr.shape[0] * 2, img_arr.shape[1] * 2, img_arr.shape[2])
    new_arr = np.full(new_size, 255, dtype=np.uint8)

    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            new_arr[2*i, 2*j] = img_arr[i, j]
            new_arr[2*i+1, 2*j] = img_arr[i, j]
            new_arr[2*i, 2*j+1] = img_arr[i, j]
            new_arr[2*i+1, 2*j+1] = img_arr[i, j]

    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def zoomout():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    x, y = img.size
    new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
    r = [0, 0, 0, 0]
    g = [0, 0, 0, 0]
    b = [0, 0, 0, 0]

    for i in range(0, int(x/2)):
        for j in range(0, int(y/2)):
            r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
            r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
            r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
            r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
            new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))

    new_img = new_arr.convert('RGB')  # Convert to RGB mode
    new_img.save("static/img/img_now.jpg")


def move_left():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    # Move to the left by shifting columns to the right
    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_right():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    # Move to the right by shifting columns to the left
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_up():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    # Move up by shifting rows downwards
    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[:-50, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_down():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    # Move down by shifting rows upwards
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img).astype(np.uint16)
    img_arr = np.clip(img_arr + 100, 0, 255)
    new_arr = img_arr.astype(np.uint8)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img).astype(np.int16)
    img_arr = np.clip(img_arr - 100, 0, 255)
    new_arr = img_arr.astype(np.uint8)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_multiplication():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    img_arr = np.clip(img_arr * 1.25, 0, 255)
    new_arr = img_arr.astype(np.uint8)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_division():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    img_arr = np.clip(img_arr / 1.25, 0, 255)
    new_arr = img_arr.astype(np.uint8)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def convolution(img, kernel):
    h_img, w_img, _ = img.shape
    h_kernel, w_kernel = kernel.shape
    pad_height = h_kernel // 2
    pad_width = w_kernel // 2
    padded_img = np.pad(img, ((pad_height, pad_height),
                        (pad_width, pad_width), (0, 0)), mode='constant')

    new_img = np.zeros_like(img)

    for y in range(h_img):
        for x in range(w_img):
            for c in range(3):  # Iterate over channels (RGB)
                roi = padded_img[y:y+h_kernel, x:x+w_kernel, c]
                weighted_sum = np.sum(roi * kernel)
                new_img[y, x, c] = np.clip(weighted_sum, 0, 255)

    return new_img


def edge_detection():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img, dtype=int)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr.astype(np.uint8))
    new_img.save("static/img/img_now.jpg")


def blur():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img, dtype=int)
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr.astype(np.uint8))
    new_img.save("static/img/img_now.jpg")


def sharpening():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img, dtype=int)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr.astype(np.uint8))
    new_img.save("static/img/img_now.jpg")


def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr[:, :, 0].flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    else:
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        data_r = Counter(r)
        data_g = Counter(g)
        data_b = Counter(b)

        plt.bar(list(data_r.keys()), data_r.values(), color='red')
        plt.savefig(f'static/img/red_histogram.jpg', dpi=300)
        plt.clf()

        plt.bar(list(data_g.keys()), data_g.values(), color='green')
        plt.savefig(f'static/img/green_histogram.jpg', dpi=300)
        plt.clf()

        plt.bar(list(data_b.keys()), data_b.values(), color='blue')
        plt.savefig(f'static/img/blue_histogram.jpg', dpi=300)
        plt.clf()


def df(img):
    values = [0] * 256
    if len(img.shape) == 2:  # Grayscale image
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                values[img[i, j]] += 1
    elif len(img.shape) == 3:  # RGB image
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    values[img[i, j, k]] += 1
    return values


def cdf(hist):
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1] + hist[i]
    # Now we normalize the histogram
    cdf_normalized = [ele * 255 / cdf[-1] for ele in cdf]
    return cdf_normalized


def histogram_equalizer():
    img = cv2.imread('static/img/img_now.jpg', 0)  # Load image as grayscale
    hist = df(img)  # Compute histogram
    my_cdf = cdf(hist)  # Compute cumulative distribution frequency
    # Use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    cv2.imwrite('static/img/img_now.jpg', image_equalized)
    # Plot and save histogram after equalization
    plt.hist(image_equalized.flatten(), bins=256, range=[0, 256], color='gray')
    plt.savefig('static/img/equalized_histogram.jpg', dpi=300)


def threshold(lower_thres, upper_thres):
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    if len(img_arr.shape) == 3:  # Color image
        condition = np.logical_and.reduce((img_arr[:, :, 0] >= lower_thres[0], img_arr[:, :, 0] <= upper_thres[0],
                                           img_arr[:, :, 1] >= lower_thres[1], img_arr[:,
                                                                                       :, 1] <= upper_thres[1],
                                           img_arr[:, :, 2] >= lower_thres[2], img_arr[:, :, 2] <= upper_thres[2]))
    else:  # Grayscale image
        condition = np.logical_and(
            img_arr >= lower_thres, img_arr <= upper_thres)
    img_arr[condition] = 255
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")
