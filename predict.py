"""
download the checkpoints here
mkdir -p VL-T5/snap/pretrain/VLT5
gdown.download('https://drive.google.com/uc?id=100qajGncE_vc4bfjVxxICwz3dwiAxbIZ', 'VL-T5/snap/pretrain/VLT5/Epoch30.pth')
"""

import sys
sys.path.insert(0, 'VL-T5/src')
sys.path.insert(0, 'VL-T5')

import tempfile
import PIL.Image
import torch
import numpy as np

from cog import BasePredictor, Path, Input

from param import parse_args
from vqa import Trainer
from tokenization import VLT5TokenizerFast
from inference.processing_image import Preprocess
from inference.visualizing_image import SingleImageViz
from inference.modeling_frcnn import GeneralizedRCNN
from inference.utils import Config, get_data


class Predictor(BasePredictor):
    def setup(self):
        args = parse_args(
            parse=False,
            backbone='t5-base',
            load='VL-T5/snap/pretrain/VLT5/Epoch30'
        )
        args.gpu = 0

        self.trainer = Trainer(args, train=False)

        OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
        ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"

        self.object_ids  = get_data(OBJ_URL)
        self.attr_ids  = get_data(ATTR_URL)
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)
        self.image_preprocess = Preprocess(self.frcnn_cfg)

        self.tokenizer = VLT5TokenizerFast.from_pretrained('t5-base')

    def predict(
            self,
            image: Path = Input(description="Input image."),
            question: str = Input(description="question for VQA")) -> str:

        frcnn_visualizer = SingleImageViz(str(image), id2obj=self.object_ids, id2attr=self.attr_ids)

        images, sizes, scales_yx = self.image_preprocess(str(image))

        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding='max_detections',
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors='pt'
        )

        # add boxes and labels to the image
        frcnn_visualizer.draw_boxes(
            output_dict.get("boxes"),
            output_dict.get("obj_ids"),
            output_dict.get("obj_probs"),
            output_dict.get("attr_ids"),
            output_dict.get("attr_probs"),
        )

        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")

        input_ids = self.tokenizer(question, return_tensors='pt', padding=True).input_ids
        batch = {}
        batch['input_ids'] = input_ids
        batch['vis_feats'] = features
        batch['boxes'] = normalized_boxes

        result = self.trainer.model.test_step(batch)

        return result['pred_ans'][0]
