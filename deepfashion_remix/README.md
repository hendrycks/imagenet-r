# DeepFashion2 Remixed

Code used to reproduce the results in [The Many Faces of Robustness](arxiv.com).

## Installation

```bash
pip install torch torchvision
```

DeepFashion2 can be downloaded from [here](https://github.com/switchablenorms/DeepFashion2). We provide a dataloader that convert the labels into multilabel task, just need to update the paths in train.py and validate.py to point to the location extracted. By default we store it in the folder above this repo as ".."

To stylize the dataset we use this repo [stylize-dataset](https://github.com/bethgelab/stylize-datasets).  We use the default settings to create 1 extra stylization per image.

We use the BiT-M-R50x1 model which can be downloaded from [here](https://github.com/google-research/big_transfer).

## Usage

Train the model

```python
python3 train.py --scales 2 --occlusion 2 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b 100 --arch resnet50 --augmix 10 --disc trainonlyaugmixsev102221
```

Evaluate the model. Provide the model name (the --disc argument) and the architecture.

```python
bash eval.sh trainonlyaugmixsev102221 resnet50
```

Requires updating the paths within `train.py` and `validate.py` to point to where the deepfashion2 dataset is located.


### Replicate experiments in the paper

```python
python3 train.py -b 100 --arch resnet50 --augmix 10 --disc trainonlyaugmixsev102221
python3 train.py -b 100 --arch resnet50 --cutout --disc trainonlyranderasure2221
python3 train.py -b 100 --arch resnet50 --speckle --disc trainonlyspeckle2221
python3 train.py -b 100 --arch resnet50 --stylize -e 40 --disc trainonlystylize2221
python3 train.py -b 100 --arch resnet50 --dataset deepaugment -e 28 --disc trainonlydeepaugment2221
python3 train.py -b 100 --arch se --disc trainonly2221
python3 train.py -b 100 --arch resnet152 --disc trainonly2221

bash eval.sh trainonlyaugmixsev102221 resnet50
....
bash eval.sh trainonly2221 resnet152
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
