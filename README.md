# Uni-GroundDiff 
The official implementation of the Uni-GroundDiff  paper in PyTorch.

## Revisiting Diffusion Models for Unified Zero-Shot Text-Guided Grounding in Images and Videos
<img width="1089" height="600" alt="image" src="https://github.com/user-attachments/assets/938ef0e7-84ca-40f4-8c20-8f083722ce4e" />

## News
* [2026-07-19] Code is released.

## Installation

### Requirements

- Python 3.8.18
- Numpy
- Pytorch 2.4.1
- detectron2 0.3.0
- GroundDino

1. Install the packages in `requirements.txt` via `pip`:
```shell
pip install -r requirements.txt
```
2. cd segment-anything-third-party && pip install -e . && cd ..

3. put SAM pretrained model https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth into ./segment-anything

4. put SAM2 checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt into ./sam2/checkpoints/

## Datasets

## **PNG**

1. Download the 2017 MSCOCO Dataset from its [official webpage](https://cocodataset.org/#download). You will need the train and validation splits' images and panoptic segmentations annotations.

2. Download the Panoptic Narrative Grounding Benchmark from the PNG's [project webpage](https://bcv-uniandes.github.io/panoptic-narrative-grounding/#downloads). Organize the files as follows:

```
datasets
|_coco
    |_ train2017
    |_ val2017
    |_ panoptic_stuff_train2017
    |_ panoptic_stuff_val2017
    |_annotations
        |_ png_coco_train2017.json
        |_ png_coco_val2017.json
        |_ panoptic_segmentation
        |  |_ train2017
        |  |_ val2017
        |_ panoptic_train2017.json
        |_ panoptic_val2017.json
        |_ instances_train2017.json
```

## Inference

1. generate attention map by six GPUs
    ```
        bash generate_diffusion_mask_png.sh
    ```
2. generate SAM candidate mask.
    ```
        bash generate_sam_mask_png.sh
    ```
3. evaluate on PNG dataset
    ```
        bash eval_png.sh
    ```

## **Ref-Davis17**

Organize the files as follows:

```
datasets
|_ ref-davis
     |_ train
        |_ JPEGImages
        |_ Annotations
        |_ meta.json
     |_ valid
        |_ JPEGImages
        |_ Annotations
        |_ meta.json
     |_ meta_expressions
        |_ train
        |_ valid
     |_ ImageSets
     |_ davis_text_annotations
     |_ convert_davis_to_ytvos.py
```

## Inference

1. generate attention map by six GPUs

   ```
       bash generate_diffusion_mask_davis17.sh
   ```

2. generate SAM candidate mask.

   ```
       bash generate_sam_mask_davis17.sh
   ```

3. evaluate on PNG dataset

   ```
       bash eval_davis17.sh
   ```

## VICAS

1. Download the VICAS Dataset from its [official webpage] https://github.com/Ali2500/ViCaS
2. Organize the files as follows:

```
  datasets
  |_ ViCaS
     |_ frames
     |  |_ <video_id>
     |     |_ 00000.jpg
     |     |_ 00001.jpg
     |     |_ ...
     |_ annotations
     |  |_ v1.0
     |     |_ <video_id>.json
     |     |_ ...
```

## Inference

1. generate attention map by six GPUs

   ```
       bash generate_diffusion_mask_VICAS.sh
   ```

2. generate SAM candidate mask.

   ```
       bash generate_sam_mask_VICAS.sh
   ```

3. evaluate on VICAS dataset

   ```
       bash eval_VICAS.sh
   ```
