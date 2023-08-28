
import io
import logging
import os
from typing import List, Tuple

import numpy as np
import onnxruntime
from PIL import Image as pil_image


def load_img(
    path, grayscale=False, color_mode="rgb", target_size=(256,256), interpolation="nearest"
):
    img = pil_image.open(path)

    if color_mode == "grayscale":
        if img.mode != "L":
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    
    width_height_tuple = (target_size[1], target_size[0])
    img = img.resize(width_height_tuple, resample=0)
    return img


def img_to_array(img, data_format="channels_last", dtype="float32"):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format: %s" % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == "channels_first":
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == "channels_first":
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError("Unsupported image shape: %s" % (x.shape,))
    return x


def load_images(image_paths, image_size, image_names):
    
    loaded_images = []
    loaded_image_paths = []

    for i, img_path in enumerate(image_paths):
        try:
            image = load_img(img_path, target_size=image_size)
            image = img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(image_names[i])
        except Exception as ex:
            logging.exception(f"Error reading {img_path} {ex}", exc_info=True)

    return np.asarray(loaded_images), loaded_image_paths


class Classifier:

    def __init__(self):
        """
        model = Classifier()
        """
        self.nsfw_model = onnxruntime.InferenceSession('classifier_model.onnx')
        print("Model Loaded")


    def classify(
        self,
        img : np.ndarray,
        batch_size : int = 4,
        categories=["unsafe", "safe"],
    ):
       
        
        preds = []
        model_preds = []
        _model_preds = self.nsfw_model.run(
            [self.nsfw_model.get_outputs()[0].name],
            {self.nsfw_model.get_inputs()[0].name: [img]},
        )[0]
        
        model_preds.append(_model_preds)
        preds += np.argsort(_model_preds, axis=1).tolist()
        print(preds)
        

        probs = []
        for i, single_preds in enumerate(preds):
            single_probs = []
            for j, pred in enumerate(single_preds):
                single_probs.append(
                    model_preds[int(i / batch_size)][int(i % batch_size)][pred]
                )
                preds[i][j] = categories[pred]

            probs.append(single_probs)

        return preds, probs


if __name__ == "__main__":
    NSFW = Classifier()
    img_path = ["imgs/1.png"]
    image = load_img(img_path[0])
    image = img_to_array(image)
    image /= 255    
    preds, probs = NSFW.classify(np.asarray(image))

    print(preds, probs)

    images_preds = {}

  
    loaded_image_path = 0

    images_preds[loaded_image_path] = {}
    for _ in range(len(preds[0])):
        images_preds[loaded_image_path][preds[0][_]] = float(probs[0][_])

    print(images_preds)
