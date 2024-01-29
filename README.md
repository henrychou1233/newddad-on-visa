## Requirements
This repository is implemented and tested on Python 3.8 and PyTorch 1.13.
To install requirements:

```setup
pip install -r requirements.txt
```

## Train and Evaluation of the Model
For tarining the denoising UNet, run:

```train
python main.py --train True
```

In the config.yaml file you can change the setting for trainnig. By chaning category, one can train the model on different categories.

For fine tuning the feature extractor, run:

```domain_adaptation
python main.py --domain_adaptation True
```
Note that in config.yaml, hyperparameter 'DA_epochs' determines the number of iteration for fine tuning. Depending on the number of trainin dataset and complexity of the category chagning this hyperparameter between 0 to 3 my help enormously.

For evaluation and testing the model, run:
```eval
python main.py --eval True
```
While we find 'w=4' a suitable number for reconstruction, increasing and decreasing this hyperparameter helps for better reconstruction.
}
```
