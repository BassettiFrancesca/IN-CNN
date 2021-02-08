import torch
import torchvision
import torchvision.transforms as transforms
import cnn


def test():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    batch_size = 4
    num_workers = 2

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = cnn.Net().to(device)
    PATH = './mnist_net.pth'
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

    for i in range(10):
        acc = 100 * n_class_correct[i] / n_class_samples[i]
        print('Accuracy of %s: %.3f %%' % (classes[i], acc))

    print('Accuracy of the network on the 10000 test images: %.3f %%' % (100 * correct / total))
