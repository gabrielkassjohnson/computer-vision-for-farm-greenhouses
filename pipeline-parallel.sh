#!/bin/bash

# Here we are running a VIS top-view workflow over a flat directory of images

# Image names for this example look like this: cam1_16-08-06-16:45_el1100s1_p19.jpg

/home/a/Desktop/plantcv/plantcv-workflow.py \
-d ./dataset/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Stacks/Ara2013-Canon/stack_08 \
-j ./pipeline.json \
-p ./pipeline.py \
-i ./output-images \
-f timestamp \
-t png \
-T 1 \
-w  \
#-a filename \
#-s %H-%M-%S \
#-l _ \
#--other_args="--background ./background/08-12-17.jpg" \
#--other_args="--debug print" 
