import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import copy

class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
            
    def __len__(self):
            return len(self.samples)
    
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]
    
def load_dataset(dataset_name, batch_size):
    if dataset_name == "cifar10":
        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        #if is_train:
        trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
        #train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
        #else:
        testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
        #test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
        classes_count = 10
        dummy_input = torch.randn(1, 3, 32, 32)
       
    elif dataset_name == "cifar100":
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        #if is_train:
        trainset = torchvision.datasets.CIFAR100(root='../cifar100-models/pytorch-cifar100/data', train=True, download=True, transform=transform_train)
        #train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
        #else:
        testset = torchvision.datasets.CIFAR100(root='../cifar100-models/pytorch-cifar100/data', train=False, download=True, transform=transform_test)
        #test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
        classes_count = 100
        dummy_input = torch.randn(1, 3, 32, 32)

    elif dataset_name == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        #if is_train:
        trainset = ImageNetKaggle("../data/imagenet/", "train", transform)
        #train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=False)
        #else:
        testset = ImageNetKaggle("../data/imagenet/", "val", transform)
        #test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=False, pin_memory=True)

        classes_count = 1000

    sub_train_dataset = copy.deepcopy(trainset)  # used for resetting bn statistics
    sub_train_dataset.transform = testset.transform
    if len(sub_train_dataset) > 16000:
        g = torch.Generator()
        g.manual_seed(937162211)
        rand_indexes = torch.randperm(len(sub_train_dataset), generator=g).tolist()
        rand_indexes = rand_indexes[:3000] # for alexnet and  vgg use 3000 for ftclip 1000
        
        if dataset_name == "cifar10":
            sub_train_dataset.data = [
                sub_train_dataset.data[idx] for idx in rand_indexes
            ]
            sub_train_dataset.targets = [
                sub_train_dataset.targets[idx] for idx in rand_indexes
            ]
        
        elif dataset_name == "cifar100":
            sub_train_dataset.data = [
                sub_train_dataset.data[idx] for idx in rand_indexes
            ]    
            sub_train_dataset.targets = [
                sub_train_dataset.targets[idx] for idx in rand_indexes
            ] 
        
        else:   
            sub_train_dataset.samples = [
                sub_train_dataset.samples[idx] for idx in rand_indexes
            ]

    #train_dataloader = torch.utils.data.DataLoader(
    # trainset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    sub_train_loader = torch.utils.data.DataLoader(
        dataset=sub_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    sub_train_loader = [data for data in sub_train_loader]

    data_loader_dict = {
        "train": train_loader,
        "val": valid_loader,
        "sub_train": sub_train_loader,
    }
    
    return data_loader_dict, classes_count  #, dummy_input
    
