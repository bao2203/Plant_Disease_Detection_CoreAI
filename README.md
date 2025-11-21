```markdown
# Plant Disease Detection AI 🌿

This repository contains the core AI codebase for a plant leaf disease detection application. The project is written in **Python** using **PyTorch** and is designed to classify leaves as healthy or diseased.  

---

## Folder Structure

```

PlantDetectionAI/
│
├─ data/
│   ├─ train/
│   │   ├─ healthy/
│   │   └─ diseased/
│   ├─ val/
│   │   ├─ healthy/
│   │   └─ diseased/
│   └─ test/
│       ├─ healthy/
│       └─ diseased/
│
├─ dataset.py        # Custom Dataset class
├─ model.py          # CNN model definition
├─ train.py          # Training loop
├─ evaluate.py       # Model evaluation
├─ utils.py          # Utility functions (optional)
├─ main.py           # Entry point
└─ README.md

````

---

## Installation

Create a Python virtual environment and install required packages:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install torch torchvision pillow numpy
````

*(Add any other libraries you used here, e.g., matplotlib, tqdm, etc.)*

---

## Usage

1. Prepare your dataset inside the `data/` folder as shown above.
2. Adjust hyperparameters in `main.py` if needed (batch size, learning rate, epochs, etc.).
3. Run the training:

```bash
python main.py
```

4. Evaluate the trained model using `evaluate.py`:

```bash
python evaluate.py
```

5. Once trained, the model can be integrated into a backend API (C#/.NET, Flask, FastAPI, etc.) or a mobile app (Flutter).

---

## Features

* Custom CNN (`PlantCNN`) for plant leaf classification.
* Dataset class for loading images from folder structure.
* Supports train, validation, and test splits.
* Easy to extend to more plant types or classes.

---

## Notes

* Ensure your dataset has enough images for each class for proper training.
* This code is for **research/prototype purposes**.
* Designed to be lightweight and easy to integrate into a mobile/desktop application.

---

## License

This project is open-source for educational purposes.

---

Bạn có muốn mình làm luôn không?
```
