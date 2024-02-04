This study introduces a paradigm shift in anomaly detection by adopting the diverse VisA dataset over the traditional MVTec, bringing in a variety of visual anomalies for exploration.

The research focused on enhancing the model's noise handling and refining the scoring of features and textures, thereby boosting detection accuracy. Architectural improvements in the UNET's residual structure and better upsampling and downsampling techniques have improved feature extraction efficiency and precision.

The model's complexity was balanced with generalization capabilities, incorporating L1 and L2 regularizations. Broader evaluation metrics now include the F1 score, latency, and FPS, leading to comprehensive performance assessments.

The improved model showcases competitive performance with an AUROC close to 98%, leading in F1 score, FPS, and latency, and setting new benchmarks for anomaly detection. The study paves the way for further enhancements in the field. Releasing the code at https://github.com/henrychou1233/newddad-on-visa

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
