from ultralytics import YOLO
from loader import WaymoE2E
import argparse
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def load_yolo(model_path: str = "yolo26x.pt"):
    # Load a model
    model = YOLO(model_path)
    return model

def inference_yolo(model, image):
    # Predict with the model
    results = model(image, verbose=False)  # predict on an image

    # Access the results
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Waymo directory")
    args = parser.parse_args()

    model = load_yolo()

    test_dataset = WaymoE2E(
        indexFile="index_val.pkl", data_dir=args.data_dir, n_items=5_000
    )

    for i in range(10):
        sample = test_dataset[i]
        jpeg = sample["IMAGES_JPEG"][1] # front cam
        if hasattr(jpeg, 'numpy'):
            jpeg = jpeg.numpy().tobytes()
        elif hasattr(jpeg, 'tobytes'):
            jpeg = jpeg.tobytes()
        image = Image.open(io.BytesIO(jpeg))
        results = inference_yolo(model, image)
        
        # results[0].plot() gives the numpy array with the bounding boxes rendered
        res_img = results[0].plot()
        plt.imsave(f"yolo_output_{i}.png", res_img)

    