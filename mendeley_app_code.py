from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from mendeley_hybridmodel import HybridModel   # your model file

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

classes = ['Lung_Opacity', 'Normal', 'Viral Pneumonia']

# Loading the model
model = HybridModel()
model.load_state_dict(torch.load("best_quantum_model.pth", map_location='cpu'))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    prediction = None
    image_file = None

    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image_file = file.filename

            # Load image
            img = Image.open(filepath).convert('RGB')
            x = transform(img).unsqueeze(0)

            # Prediction
            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1)

                pred_class = torch.argmax(probs, dim=1).item()
                pred_prob = probs[0][pred_class].item() * 100

                result = classes[pred_class]

                prediction = {
                    "result": result,
                    "confidence": f"{pred_prob:.2f}%"
                }

    return render_template("mendeley_index.html",
                           prediction=prediction,
                           image_file=image_file)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)