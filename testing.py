import torch
import cnn


def test(test_set, out):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 4
    num_workers = 2

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = cnn.Net(out).to(device)
    PATH = './mnist_net.pth'
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %.2f %%\n' % (100 * correct / total))

    return 100 * correct / total
