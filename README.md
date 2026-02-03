# Fake Currency Detector

![Project Banner](https://via.placeholder.com/1000x300?text=Fake+Currency+Detector)

> **Advanced AI-powered system to detect counterfeit currency notes using Deep Learning.**

This project utilizes state-of-the-art Convolutional Neural Networks (CNNs), including MobileNet and ResNet50 Transfer Learning architectures, to classify currency notes as **Real** or **Fake**. It features a training pipeline and a modern, responsive Web GUI for easy interaction.

## ✨ Features

- **🚀 Advanced Models**: Choose between MobileNetV2, ResNet50, or a custom CNN.
- **📈 Training Pipeline**: Built-in script to train, evaluate, and save models.
- **🖥️ Modern GUI**: Beautiful, user-friendly interface with drag-and-drop support.
- **⚡ Real-time Inference**: Fast predictions powered by TensorFlow and FastAPI.
- **📊 Visualization**: Generates accuracy and loss plots during training.

## 📂 Project Structure

```
fake_currency_detector/
├── app/
│   ├── main.py              # FastAPI Application & Endpoints
│   ├── model_definitions.py # Model Architectures (MobileNet, ResNet, CNN)
│   ├── predict.py           # Inference Logic
│   ├── preprocess.py        # Image Preprocessing Utilities
│   └── schemas.py           # Pydantic Models
├── data/                    # Dataset Directory (Train/Test)
├── model/                   # Trained Models & Plots
├── static/                  # Frontend Assets (HTML, CSS, JS)
├── train.py                 # Training Script
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/fake_currency_detector.git
    cd fake_currency_detector
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Training the Model

You can train the model using your own dataset. The dataset should be organized into `Train` and `Test` folders within a `data` directory.

**Command:**
```bash
python train.py --data ./data --model mobilenet --epochs 10
```

**Arguments:**
- `--data`: Path to the dataset directory (must contain `Train` and `Test` subfolders).
- `--model`: Architecture to use (`mobilenet`, `resnet`, or `cnn`). Default: `mobilenet`.
- `--epochs`: Number of training epochs. Default: `5`.

**Output:**
- The trained model will be saved to `model/{model_type}_model.h5`.
- Accuracy plots will be saved to `model/{model_type}_accuracy.png`.

## 🚀 Running the Application

1.  **Start the Server**
    ```bash
    uvicorn app.main:app --reload
    ```

2.  **Open the GUI**
    Navigate to [http://localhost:8000](http://localhost:8000) in your browser.

3.  **Use the Detector**
    - Drag and drop an image of a currency note.
    - Click "Analyze Currency".
    - View the prediction and confidence score.

## 📊 Dataset Format

Ensure your data directory looks like this:

```
data/
├── Train/
│   ├── Real/
│   │   ├── image1.jpg
│   │   └── ...
│   └── Fake/
│       ├── image1.jpg
│       └── ...
└── Test/
    ├── Real/
    │   └── ...
    └── Fake/
        └── ...
```

## 📜 License

This project is licensed under the MIT License.
