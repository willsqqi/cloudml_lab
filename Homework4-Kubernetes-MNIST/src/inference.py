"""Serve a page that predicts a digit selected by its MNIST test-image ID."""

import base64
from io import BytesIO
import os
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision.datasets.mnist import read_image_file

from model import Net, transform


def create_app():
    app = Flask(__name__)
    data_dir = os.environ.get("DATA_DIR", "data")
    model_path = os.environ.get("MODEL_PATH", "model/mnist_cnn.pt")

    # Startup fails if the trained model or test data is missing.
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    # MNIST's dataset class checks both splits; serving needs only test images.
    test_images = read_image_file(Path(data_dir) / "MNIST/raw/t10k-images-idx3-ubyte")
    last_id = len(test_images) - 1

    @app.get("/")
    def index():
        return render_template("index.html", last_id=last_id, image_id=0)

    @app.post("/predict")
    def predict():
        try:
            image_id = int(request.form.get("image_id", ""))
            if not 0 <= image_id <= last_id:
                raise ValueError
        except ValueError:
            return render_template(
                "index.html", last_id=last_id, image_id="",
                error=f"Enter a whole-number image ID from 0 to {last_id}.",
            ), 400

        image = Image.fromarray(test_images[image_id].numpy())
        with torch.inference_mode():
            digit = model(transform(image).unsqueeze(0)).argmax(dim=1).item()
        png = BytesIO()
        image.save(png, format="PNG")
        image_base64 = base64.b64encode(png.getvalue()).decode("ascii")
        print(f"image_id={image_id} predicted_digit={digit}", flush=True)
        return render_template(
            "index.html", last_id=last_id, image_id=image_id,
            digit=digit, image_base64=image_base64,
        )

    @app.get("/healthz")
    def health():
        # Routes are served only after model/data loading succeeds above.
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8080)
