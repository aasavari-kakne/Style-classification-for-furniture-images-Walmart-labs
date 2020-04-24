# Contrastive-Predictive-Coding-for-Image-Recognition-in-PyTorch
PyTorch Implementation of Contrastive Predictive Coding for image recognition, based on reference 2 and 3. 

### See more details in this [blog post](https://mf1024.github.io/2019/05/27/contrastive-predictive-coding/).

# Dependecies:
Run ```pip3 install -r requirements.txt``` and then 
1. Install PyTorch using following instructions. 
```
pip3 install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```
or 
```
conda install -c pytorch torchvision
```
2. Always use pip3 and python3 commands. 



# script 1 : imagenet_dataset.py

## class ImagenetDataset 
This class inherits from torch.utils.data.Dataset (reference 1). 

### class attributes (other than the parent class):
1) data_path: str which is a path of a folder inside which there is a folder with name matching the name of the class, for each class. Inside every class folder, there are images belonging to that class.
2) is_class_limited: bool denoting if we know the number of classes.
3) classes: list of dictionaries holding class id and class name (same as class folder name).
4) num_classes: number of classes.
5) image_list: list of dictionaries in which first item is dictionary of class id andclass name where as later two items are image path and image name.
6) img_idxes: numpy array from 0 to number of images - 1.

### Important class methods are as follows - 
1) getitem: if the mode of the image is 'L', then converts the image into grayscale pytorch tensor, otherwise converts the image into rgb pytorch tensor, by randomised crop of size 256x256. (a center crop looks like a better choice for our images.)

### other important functions -
1) get_imagenet_datasets: returns ImagenetDataset objects for training and testing respectively. Later, these objects will be fed into PyTorch dataloader.  

        

# Resources for using icme-gpu:
## Job description sprint to submit a job to slurm:
Name this file "submit.sh" containing following lines and run "sbatch submit.sh" in terminal. This will send you an email when the job is done or failed. 
```
#!/bin/bash
#SBATCH --job-name=gputest1
# Get email notification when job finishes or fails
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=<sunetid>@stanford.edu
# Define how long you job will run d-hh:mm:ss
#SBATCH --time 02:00:00
# GPU jobs require you to specify partition
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --mem=16G
# Number of tasks
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
```
### Other slurm resources (check reference 4) :
1) squeue -u <user_name> : checks status of the jobs for this user.
2) squeue <job_id> : checks status of a particular job.
3) slurm-<job_id>.out : log of output.

# Regarding git:

## Pulling a git repo on local machine:
In terminal run the following commands
1. $mkdir dir
2. $cd dir
3. $git init
4. $git clone <link_to_github_repo>

## Pulling a git repo on icme-gpu server:
In terminal, run the following commands
1. $ssh -Y <sunet_id>@icme-gpu.stanford.edu
2. $git clone <link_to_github_repo>

## Other useful git commands:
1. git status - helps to see where the local branch is compared to master. 
2. git diff - helps see what changes have been made locally to the files in the repo. 
3. git commit -am "message" - commits local changes
4. git push origin master - push local changes to master
5. git pull origin master - pull master to local.

# References:
1) https://pytorch.org/docs/stable/data.html - see torch.utils.data.Dataset
2) https://arxiv.org/abs/1807.03748
3) https://arxiv.org/abs/1905.09272
4) https://srcc.stanford.edu/sge-slurm-conversion



# Questions, comments and notes:
## Imagenet_datasets.py:
1) Line 81 in imagenet_datsets.py: changed img1 to img, because img1 was not being used in rest of the script. 
2) I have tried changing tensorflow tensor to numpy array and then to torch tensor. Hopefully it will work. 
3) When running training, comment out everything after the functions. It is just for sanity check that the datasets are what we need. 
