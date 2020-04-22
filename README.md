# Contrastive-Predictive-Coding-for-Image-Recognition-in-PyTorch
PyTorch Implementation of Contrastive Predictive Coding for image recognition, based on reference 2 and 3. 

### See more details in this [blog post](https://mf1024.github.io/2019/05/27/contrastive-predictive-coding/).

# Dependecies:
Run ```pip3 install -r requirements.txt``` and then install PyTorch using following instructions. 
```
pip3 install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```
or 
```
conda install -c pytorch torchvision
```


# script 1 : imagenet_dataset.py
## class ImagenetDataset 
This class inherits from torch.utils.data.Dataset (reference 1). 
### Important class methods are as follows - 
1) Constructor: data_path is path of a folder inside which there is a folder with name matching the name of the class, for each class. Inside every class folder, there are images belonging to that class.
2) getitem: if the mode of the image is 'L', then converts the image into grayscale pytorch tensor, otherwise converts the image into rgb pytorch tensor, by randomised crop of size 256x256. (a center crop looks like a better choice for our images.)
### other important functions -
1) get_imagenet_datasets: returns ImagenetDataset objects for training and testing respectively. Later, these objects will be fed into PyTorch dataloader.  

        

# References:
1) https://pytorch.org/docs/stable/data.html - see torch.utils.data.Dataset
2) https://arxiv.org/abs/1807.03748
3) https://arxiv.org/abs/1905.09272
