import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

def return_data(name, dset_dir, batch_size):
    
    if 'MNIST' in name :
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),])
        root = os.path.join(dset_dir,'MNIST')
        train_kwargs = {'root':root,'train':True,'transform':transform,'download':True}
        test_kwargs = {'root':root,'train':False,'transform':transform,'download':False}
        dset = MNIST
    
    elif 'CIFAR10' in name:
        transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

        root = os.path.join(dset_dir,'CIFAR10')
        train_kwargs = {'root':root,'train':True,'transform':transform,'download':True}
        test_kwargs = {'root':root,'train':False,'transform':transform,'download':False}
        dset = CIFAR10

    elif 'CIFAR100' in name:
        transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

        root = os.path.join(dset_dir,'CIFAR100')
        train_kwargs = {'root':root,'train':True,'transform':transform,'download':True}
        test_kwargs = {'root':root,'train':False,'transform':transform,'download':False}
        dset = CIFAR10


    else : raise UnknownDatasetError()

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1,
                                drop_last=True)
    train_bootstrap_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1,
                                drop_last=True)

    test_data = dset(**test_kwargs)
    test_loader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
    
    test_bootstrap_loader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1,
                                drop_last=True)

    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['train_bootstrap'] = train_bootstrap_loader
    data_loader['test'] = test_loader
    data_loader['test_bootstrap'] = test_bootstrap_loader

    return data_loader

if __name__ == '__main__' :
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--dset_dir', default='datasets', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
