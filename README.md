# flora-vision
FloraVision: AI-powered flower identification system focused on Nepali flowers, using CNN with scientific and medicinal insights.
# FloraVision – Flower Identification System

FloraVision is an AI-powered web application that identifies flowers from images and provides detailed information such as scientific names, medical uses, and additional insights.

It combines deep learning with a simple web interface to create an interactive plant recognition experience.

---

## 🚀 Features

* 📷 Upload flower images
* 🧠 Deep Learning-based classification (Custom CNN)
* 🌿 Displays:

  * Flower Name
  * Scientific Name
  * Medical Uses
  * Extra Details
* 💾 Stores uploaded images for preview
* ⚡ Fast and lightweight Flask backend

---

## 🛠 Tech Stack

* **Frontend:** HTML, CSS (Jinja Templates)
* **Backend:** Flask (Python)
* **Machine Learning:** PyTorch
* **Image Processing:** PIL, torchvision
* **Data Storage:** CSV

---

## 📁 Project Structure

```
FloraVision/
│
├── static/
│   └── uploads/          # Uploaded images (ignored in Git)
│
├── templates/
│   ├── index.html
│   ├── home.html
│   ├── identify.html
│   └── about.html
│
├── app.py                # Main Flask app
├── model.py              # CNN architecture
├── training.py           # Model training script
├── evaluate.py           # Evaluation script
├── flowers.csv           # Flower information dataset
├── class_to_idx.pth      # Class mapping
├── best_model.pth        # Trained model (ignored)
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/floravision.git
cd floravision


### 2. Install Dependencies

### 3. Run the Application

```bash
python app.py


---

### 4. Open in Browser

```
http://127.0.0.1:5000/
```

---

## 🧠 How It Works

1. User uploads a flower image
2. Image is preprocessed (resize + normalization)
3. CNN model predicts flower class
4. Prediction is mapped to class name
5. Flower details are fetched from CSV
6. Results are displayed on the web page

---

## 📊 Model Details

* Custom CNN Architecture
* Input Size: 128 × 128
* Output Classes: 10 flower categories
* Framework: PyTorch

---

## 🚫 Ignored Files

To keep the repository lightweight, the following are excluded:

* `.pth` model files
* Dataset folders
* Uploaded images
* Cache files

---

## 🌱 Future Improvements

* 🌍 Deploy online (Render / Railway)
* 📱 Mobile-friendly UI
* 🔍 More flower classes
* 🎯 Higher accuracy model
* 🌐 API integration for plant databases

---

##  Author

**Sharada Chaudhary**
Engineering Student | AI + UI/UX Enthusiast

