# Visual_intelligence_project
## Introduction
This project focuses on the task of object substitution in RGB images. Concretely, we target the task of creating a fully automate data augmentation system that is able to produce new, realistic images given multi-person scenarios, containing full human-bodies. To approach this problem we propose a pipeline based on, first of all, a segmentation module that is able to segment and detect the human bodies that are present in the image. We then use a pose-transfer model to generate realistic replacements for each human body, which is then complemented with a fusion and in-painting module which produces the final image. 

## Running it...
### Create the environment
First of all, the users should create a Python 3.8 environment, and execute the following command to donwload the corresponding dependencies:

```
    pip install -r requirements.txt
```

### Download the necessary data and checkpoints
In order to execute this script, the user should check the [readme.md](./human_body_generation/README.md) file, which contains specific instructions on how to donwload the checkpoints of the pre-trained generative model, as well as the corresponding datasets.

### Run the script
Finally, simply run the following command to execute the main pipeline.

```
    python main.py
```

Observe that in this script, one can change the input image by simply changing its corresponding path.

