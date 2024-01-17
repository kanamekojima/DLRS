# DLRS
DLRS is a deep learning-based segmentation method for whole slide images (WSIs) of renal biopsy specimens. It is comprised of two deep learning models: DLRS-tissue and DLRS-nucleus. DLRS-tissue segments WSIs into non-tissue areas, glomeruli, tubules, interstitia, and arteries, while DLRS-nucleus detects nuclei in the interstitium. We recommend further filtering of nuclei not in the interstitium using the estimated interstitium from DLRS-tissue.

## Installation
### Requirements: Python 3.8 or higher
```commandline
git clone https://github.com/kaname/kojima/DLRS.git
cd DLRS
wget https://media.githubusercontent.com/media/kanamekojima/DLRS/main/checkpoints/DLRS_tissue.pth -O checkpoints/DLRS_tissue.pth
wget https://media.githubusercontent.com/media/kanamekojima/DLRS/main/checkpoints/DLRS_nucleus.pth -O checkpoints/DLRS_nucleus.pth
python3 -m pip install -r requirements.txt
```
Note: OpenCV libraries are required for installation of opencv-python.

DLRS also requires PyTorch. For installation of PyTorch, please visit the [PyTorch installation page](https://pytorch.org/get-started/locally/).

## Example Usage
### Segmentation of Non-Tissue Areas, Glomeruli, Tubules, Interstitia, and Arteries
To generate a slide image with segmentation annotation, use the pretrained model `DLRS_tissue.pth` with the following command:
```sh
python3 scripts/inference.py \
  --image-file IMAGE_FILE \
  --output-file OUTPUT_FILE \
  --checkpoint checkpoints/DLRS_tissue.pth \
  --segmentation-type tissue \
  --batch-size 100
```
Note: The input slide image should be in a format compatible with OpenCV.

### Detection of Nuclei in Interstitium
To annotate detected nuclei on a slide image, use the pretrained model `DLRS_nucleus.pth` with the following command:
```sh
python3 scripts/inference.py \
  --image-file SLIDE_IMAGE_FILE \
  --output-file OUTPUT_FILE \
  --checkpoint checkpoints/DLRS_nucleus.pth \
  --segmentation-type nucleus \
  --batch-size 100
```
Note: The input slide image should be in a format compatible with OpenCV.

## Training
To train the deep learning model for DLRS with your own data, first prepare annotated data as images where each pixel value corresponds to a class number for each slide. Then, create files `TRAIN_DATA_LIST_FILE` and `VALIDATION_DATA_LIST_FILE` that list pairs of slide image files and their corresponding annotation files, with each pair on a separate line. Use these files for training and validation datasets, respectively. Execute the following command to start the training process:
```sh
python3 scripts/train.py \
  --train-list TRAIN_DATA_LIST_FILE \
  --validation-list VALIDATION_DATA_LIST_FILE \
  --output-dir results \
  --output-basename train \
  --num-classes 5 \
  --train-batch-size 10 \
  --validation-batch-size 50 \
  --segmentation-type tissue
```

## LICENSE
Scripts in this repository are licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## CONTACT
Developer: Kaname Kojima, Ph.D.

E-mail: kojima [AT] megabank [DOT] tohoku [DOT] ac [DOT] jp
