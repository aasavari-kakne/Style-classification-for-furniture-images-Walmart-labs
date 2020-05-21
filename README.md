# Style classification for furniture images | Walmart labs
PyTorch Implementation of Contrastive Predictive Coding for image recognition, based on reference 2 and 3. 
See more details in this [blog post](https://mf1024.github.io/2019/05/27/contrastive-predictive-coding/).

## Dependecies:
Always use pip3 and python3 commands. 
1. Run ```pip3 install -r requirements.txt``` 
2. Install PyTorch using following instructions. <br>
``` pip3 install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html ``` <br>
or <br>
``` conda install -c pytorch torchvision ```

## Data Cleaning:
PyTorch can throw errors for images in "untagged" folder. Do -
1) clean all ".gif" and ".png files"
2) for ".jpg", refer to "corrupted_images.xlsx"

## Loading weights from checkpoints
In main.py on line 74 and 75, specify the path to .pt files containing weights. 

## Changing size of training and testing data
In imagenet_dataset.py on line 139, 140 and 141 change parameter "train_split". This parameter decides what percentage of the images in a paerticular directory (train, test , validation) will used to create the correspaonding dataset. 

> Ideally, we should use full training and testing data. 
> We are not using the validation dataset for now. In future, we can merge that into training data. 

## Learning rates
In classificator_training.py on line 33 and 34, edit the learning rates for encoder and classifier respectively. 
Current learning rates are: 1e-6 for encoder and 1e-2 for classifier. 


## References:
1) https://pytorch.org/docs/stable/data.html - see torch.utils.data.Dataset
2) https://arxiv.org/abs/1807.03748
3) https://arxiv.org/abs/1905.09272
4) https://srcc.stanford.edu/sge-slurm-conversion
