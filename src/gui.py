# ==========================================================
# AI Background Remover
# Copyright (C) 2026 Saw it See had
# Licensed under the MIT License
# ==========================================================

import os
import numpy as np
import onnxruntime as ort
from PIL import Image

class Engine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.init_model()

    def init_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        try:
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            raise Exception(f"Failed to load ONNX model: {e}")

    def remove_background(self, image_path):
        if not self.session:
            raise Exception("Model has not been initialized!")

        try:
            orig_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise Exception(f"Cannot open image: {e}")

        w, h = orig_image.size
        input_image = orig_image.resize((224, 224), Image.Resampling.BILINEAR)

        img_data = np.array(input_image).astype(np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1)) 
        img_data = np.expand_dims(img_data, axis=0) 

        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: img_data})
        mask = output[0][0][0] 
        mask = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask, mode='L')

        mask_pil = mask_pil.resize((w, h), Image.Resampling.BILINEAR)
        mask_pil = mask_pil.point(lambda p: 255 if p > 128 else 0)

        result_image = orig_image.copy()
        result_image.putalpha(mask_pil)

        return result_image