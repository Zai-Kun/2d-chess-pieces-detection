# 2D chess board and pieces detection

This repository is about training a YOLO11n model for detecting 2D chess board and pieces in an image

## Usability and accuracy
It's pretty good and accurate; almost never fails but when it does, in most cases it's because either the board is too small on the image or the board and chess pieces are too different from the training data.

## Data generation
I used various chess pieces sets and boards from different websites (mainly from chess.com and lichess) and made a script to generate random chess images with the labels. There are several settings you can tweak inside of 'generate_datasets.py.'

## Training the model
To train the model, I first fine-tuned the pre-trained yolo11n.pt for about 20 epochs and then fine-tuned even more by generating a new dataset and training it for 30 epochs with optimizer=adamW. This probably isn't the best way to train it but for now, it's good.

## Usage
You can find the trained models in the releases page in both formats (pt, onnx).
