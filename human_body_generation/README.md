# Pose-Transfer
This module implements the model proposed by **Progressive Pose Attention for Person Image Generation** in **CVPR19(Oral)**. The paper is available [here](http://arxiv.org/abs/1904.03349). 

## Download checkpoints and models
In order to download all the required pre-trained models and checkpoints, just download this [zip file](https://docs.google.com/uc?export=download&id=1KRPKheU6i5pCoAcBpI8RvnIMnmMYNIEm), and unzip it under the *human_body_generation* directory.


#### Download Market1501
We provide our **dataset split files** and **extracted keypoints files** for convience. To this end, download the [zip file]() and unzip it under the human_body_generation directory, under the name of *market_data*.


## Train a model
To train the model --even though this is not necessary as the pre-trained model is provided, execute:
```
  python train.py
```

## Test the model
To test the model execute the following command:
```
    python test.py 
```

## Inference
To do inference, execute the following command:
```
    python inference.py
```

Notice that in the *inference.py* script, one can change the path of the input image.
