import cv2
from flask import render_template, Flask, request, jsonify
from transformers import ViTForImageClassification
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from werkzeug.utils import secure_filename
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

model_path = './static/models/v2/'  # Update with the path to your saved model
model = ViTForImageClassification.from_pretrained(model_path).to(device)

UPLOAD_FOLDER = './static/uploaded_images'
PATCH_FOLDER = './static/patch_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PATCH_FOLDER'] = PATCH_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PATCH_FOLDER, exist_ok=True)

last_uploaded_file_path = None
last_uploaded_patch_path = None

def crop_image(image_path, crop_size, offset_percent):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Error: Unable to load image.")
        height, width = image.shape[:2]
        crop_width, crop_height = crop_size

        start_width = int(width * offset_percent)
        start_height = int(height * offset_percent)

        end_width = start_width + crop_width
        end_height = start_height + crop_height

        cropped_image = image[start_height:end_height, start_width:end_width]

        cropped_image_path = os.path.join(app.config['PATCH_FOLDER'], 'cropped_image.jpg')
        cv2.imwrite(cropped_image_path, cropped_image)
        return cropped_image_path
    except Exception as e:
        print(f"Error in crop_image: {e}")
        raise

@app.route('/upload_image', methods=['POST'])
def upload_image():
    global last_uploaded_file_path

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            last_uploaded_file_path = filepath
            return jsonify({'message': 'File uploaded successfully', 'filepath': filepath}), 200
    except Exception as e:
        print(f"Error in upload_image: {e}")
        return jsonify({'error': f"Error during upload: {e}"}), 500

@app.route('/identify_writer', methods=['POST'])
def identify_writer():
    global last_uploaded_file_path, last_uploaded_patch_path

    try:
        if last_uploaded_file_path:
            crop_size = (256, 256)
            offset_percent = 0.35

            last_uploaded_patch_path = crop_image(last_uploaded_file_path, crop_size, offset_percent)
            print(f"Cropped image path: {last_uploaded_patch_path}")  # Debug print
            
            predicted_label, confidence_score = predict(last_uploaded_patch_path)
            return jsonify({'prediction': predicted_label, 'confidence': confidence_score}), 200
        else:
            return jsonify({'error': 'No image uploaded yet. Please upload an image first.'}), 400
    except Exception as e:
        print(f"Error in identify_writer: {e}")
        return jsonify({'error': f"Error during prediction: {e}"}), 500

def predict(image_path):
    try:
        id2label = {
            0: 'Aakefa Qaiser', 1: 'Aaminah Qaiser', 2: 'Abdur Rehman', 3: 'Anusha Hamza', 4: 'Ariba Yasmeen',
            5: 'Asher Hussain Rizvi', 6: 'Bazil Bashir Faridi', 7: 'Ebad Ali', 8: 'Farah Hussain', 9: 'Farhana Qaiser',
            10: 'Haseeb Shabbir', 11: 'Laiba Azam', 12: 'Malaika Ali', 13: 'Marium Rizvi', 14: 'Muhammad Ali Rizvi',
            15: 'Rameez Khan', 16: 'Shagufta Aleem', 17: 'Shaheer Ali Agha', 18: 'Ushna Hamza', 19: 'Usman Rizvi'
        }
        image = Image.open(image_path)
        print("Image opened successfully")  # Debug print

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_image = transform(image)

        with torch.no_grad():
            inputs = input_image.unsqueeze(0).to(device)
            outputs = model(inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, dim=1)
        predicted_label = id2label[predicted_class_idx.item()]
        confidence_score = confidence.item()
        return predicted_label, confidence_score
    except Exception as e:
        print(f"Error in predict: {e}")
        raise e


@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/get_last_uploaded_file_path', methods=['GET'])
def get_last_uploaded_file_path():
    return f'Last uploaded file path: {last_uploaded_file_path}'

if __name__ == '__main__':
    app.run(debug=True)