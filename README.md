# Beyond retinotopy: exploiting native visual representations in cortical neuroprostheses for vision loss remediation

![alt text](readme_imgs/pipeline.png)
![alt text](readme_imgs/results.png)

## Paper

Read the paper on [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.03.684808v1)

## Environment setup

```bash
./setup_environment.sh
```

## Code structure

- `LSV1M_training/`: experiments on data from the large scale V1 dataset with the spiking neural network model.
- `monkey_training/`: experiments on monkey data.
- `optogenetic_experiments/`: optogenetic experiments on the LSV1M model.

## Load models

```python
# load model trained on LSV1M dataset
from LSV1M_training.load_best_models import load_model_x
model = load_model_x(neurons_subset='0', return_dataloaders=False)  
```

```python
# load model trained on macaque dataset (need to first download and preprocess data)
from monkey_training.load_best_models import get_best_brcnn_model
model = get_best_brcnn_model(neurons_subset='0', return_dataloaders=False)  
```

## Download and preprocess monkey data
first download the data
```bash
wget --content-disposition      --user-agent="Mozilla/5.0"      https://figshare.com/ndownloader/files/40805201
```
then unzip it
```bash
unzip v1_data.zip
```
finally preprocess it for the dataloaders
```bash
python convert_img_to_npy.py
```


## LSV1M optogenetic stimulation experiments

The large-scale V1 model optogenetic stimulation experiments were run with [Mozaik](https://github.com/CSNG-MFF/mozaik) and [Mozaik Models](https://github.com/lucabaroni/mozaik-models)


### Citation

If you use this code in your research, please cite:
```bibtex
@article{Baroni2025.11.03.684808,
    author = {Baroni, Luca and Picek, Martin and Patel, Saumil and Tolias, Andreas S. and Antolik, Jan},
    title = {Beyond retinotopy: exploiting native visual representations in cortical neuroprostheses for vision loss remediation},
    elocation-id = {2025.11.03.684808},
    year = {2025},
    doi = {10.1101/2025.11.03.684808},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2025/11/04/2025.11.03.684808},
    journal = {bioRxiv}
}
```
