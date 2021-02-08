import testing
import training
import time


def mnist_cnn(train, test):

    start = time.time()

    if train:
        training.train()

    if test:
        testing.test()

    finish = time.time()

    print('Seconds passed: %.3f' % (finish - start))


if __name__ == '__main__':
    mnist_cnn(True, True)
