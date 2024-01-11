#!/bin/sh

train()
{
  python3 scripts/train.py \
    --train-list data/train/train.txt \
    --validation-list data/train/validation.txt \
    --output-dir results \
    --output-basename train \
    --num-classes 5 \
    --train-batch-size 10 \
    --validation-batch-size 50 \
    --segmentation-type tissue
}

DLRS_tissue()
{
  image_file=$1
  output_file=$2
  python3 scripts/inference.py \
    --image-file $image_file \
    --output-file $output_file \
    --checkpoint checkpoints/DLRS_tissue.pth \
    --segmentation-type tissue \
    --batch-size 100
}

DLRS_nucleus()
{
  image_file=$1
  output_file=$2
  python3 scripts/inference.py \
    --image-file $image_file \
    --output-file $output_file \
    --checkpoint checkpoints/DLRS_nucleus.pth \
    --segmentation-type nucleus \
    --batch-size 100
}
