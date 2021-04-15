import torch
import torchvision
import torchvision.transforms as transforms
import GroupDataset


def prepare_dataset(digits):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=False, num_workers=2)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, num_workers=2)

    train_indices = []

    for i, (image, label) in enumerate(train_loader):
        for j in range(len(digits)):
            if label[0] in digits[j]:
                train_indices.append(i)

    sub_train_dataset = torch.utils.data.Subset(train_set, train_indices)

    train_dataset = GroupDataset.GroupDataset(sub_train_dataset, digits)

    print(f'Size train dataset {digits}: {len(train_dataset)}')

    test_indices = []

    for i, (image, label) in enumerate(test_loader):
        for j in range(len(digits)):
            if label[0] in digits[j]:
                test_indices.append(i)

    sub_test_dataset = torch.utils.data.Subset(test_set, test_indices)

    test_dataset = GroupDataset.GroupDataset(sub_test_dataset, digits)

    print(f'Size test dataset {digits}: {len(test_dataset)}\n')

    return train_dataset, test_dataset
