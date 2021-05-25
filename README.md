# Unifying Vision-and-Language Tasks via Text Generation

* Authors: [Jaemin Cho](https://j-min.io), [Jie Lei](https://www.cs.unc.edu/~jielei/), [Hao Tan](https://www.cs.unc.edu/~airsplay/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
* [Paper](https://arxiv.org/abs/2102.02779) (To appear in [ICML 2021](https://icml.cc/Conferences/2021))

## Setup
```
# Create python environment (optional)
conda create -n vlt5 python=3.7

# Install python dependencies
pip install -r requirements.txt

# Download T5/BART backbone checkpoint
python download_backbones.py

# For MSCOCO captioning evaluation
python -c "import language_evaluation; language_evaluation.download('coco')"
```

## Code structure
```
# Store images, features, and annotations
./datasets
    COCO/
        images/
        featuers/
    VG/
        images/
        features/
    GQA/
        images/
        features/
    nlvr2/
        images/
        features/
    RefCOCO/

    ...

# Run feature extraction
./feature_extraction

# Train VL-T5
./src/
    modeling_t5.py modeling_bart.py                       <= VL-T5/VL-BART model classes
    pretrain.py, pretrain_data.py, pretrain_model.py      <= pretraining
    vqa.py, vqa_data.py vqa_model.py ...                  <= fine-tuning on downstream tasks (ex. VQA, GQA, NLVR2)
    multitask.py, multitask_data.py multiask_model.py     <= multitask learning on 7 downstream tasks
    param.py                                              <= (argparse) configuration
    tokenization.py                                       <= custom tokenizer
    utils.py, dist_utils.py                               <= utility functions
./snap/                                                   <= store weight checkpoints
./scripts/                                                <= bash scripts for pretraining and finetuning
```

## API
```python
import sys
sys.path.append('./src')

# Parse configuration
from param import parse_args
args = parse_args(
    backbone='t5-base' # Backbone architecture
    load='...' # Pretrained checkpoints (TBD)
    parse=False, # False for interactive env (ex. jupyter)
)
# Assign GPU
args.gpu = 0

# Load data loaders (TBD)
# from pretrain_data import get_loader
train_loader = []
val_loader = []
test_loader = []

# Import trainer
from pretrain import Trainer
trainer = Trainer(
    args,
    train_loader = train_loader
    val_loader = val_loader
    test_loader = test_loader
)

# Model is binded to trainer
model = trainer.model
print(model)
>>>
VLT5Pretraining(
    (shared): Embedding(...)
    (encoder): JointEncoder(...)
    ...
)
```


## Pretrained Models
- To be updated

## Dataset Preparation / Feature extraction
- Pretraining features (COCO/VG): download from [LXMERT github](https://github.com/airsplay/lxmert)
- Others: manually extract with [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention/)
- Details to be updated

## Pretraining on COCO+VG
```
# with 4 gpus
bash scripts/pretrain_VLT5.sh 4
bash scripts/pretrain_VLBart.sh 4
```

## Downstream tasks
- To be updated

### [VQA](https://visualqa.org/)

### [GQA](https://cs.stanford.edu/people/dorarad/gqa/)

### [NLVR2](http://lil.nlp.cornell.edu/nlvr/)

### [RefCOCOg](https://github.com/mjhucla/Google_Refexp_toolbox)

### [VCR](https://visualcommonsense.com/)

### [COCO Caption](https://cocodataset.org/)

### [Multi30K](https://github.com/multi30k/dataset)
