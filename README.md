# Style classification for furniture images | Walmart labs
PyTorch Implementation of Contrastive Predictive Coding for image recognition, based on reference 2 and 3. 
See more details in this [blog post](https://mf1024.github.io/2019/05/27/contrastive-predictive-coding/).

## Dependecies:
Run ```pip3 install -r requirements.txt``` and then 
1. Install PyTorch using following instructions. 
``` pip3 install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html ```
or
``` conda install -c pytorch torchvision ```
2. Always use pip3 and python3 commands. 

## Data Cleaning:
1) clean all ".gif" and ".png files"
2) for ".jpg", refer to "corrupted_images.xlsx

## Hyper parameters 
### Loading weights from checkpoints

2) Learning rates - 




## References:
1) https://pytorch.org/docs/stable/data.html - see torch.utils.data.Dataset
2) https://arxiv.org/abs/1807.03748
3) https://arxiv.org/abs/1905.09272
4) https://srcc.stanford.edu/sge-slurm-conversion
