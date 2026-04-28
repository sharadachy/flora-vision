from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from model import CustomCNN
import csv
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------- Model Setup -----------------
model = CustomCNN(num_classes=10)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
model.eval()

# ----------------- Load Class Mapping -----------------
class_to_idx = torch.load("class_to_idx.pth")
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ----------------- Load Flower Info from CSV -----------------
flower_info = {}

with open("flowers.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        flower_info[row['flower_name']] = {
            "Scientific Name": row['scientific_name'],
            "Medical Uses": row['medical_uses'],
            "Extra Details": row['extra_details']
        }

# ----------------- Image Transform -----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/identify")
def identify():
    return render_template("identify.html")

@app.route("/about")
def about():
    return render_template("about.html")


# ----------------- Routes -----------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error="Please select an image")

    try:
        # ✅ Save Image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # ✅ Open and Process Image
        img = Image.open(filepath).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            pred_index = torch.argmax(outputs, dim=1).item()
            predicted_class = idx_to_class[pred_index]

        info = flower_info.get(predicted_class, None)
        flower_name_display = predicted_class.replace("_", " ").title()

        return render_template(
            'index.html',
            flower_name=flower_name_display,
            flower_info=info,
            image_file=filename   # 🔥 Pass image to HTML
        )

    except Exception as e:
        return render_template('index.html', error="Error processing image")


# ----------------- Run App -----------------
if __name__ == '__main__':
    app.run(debug=True)
