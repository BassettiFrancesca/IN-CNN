import testing
import training
import time
import prepare_dataset


def mnist_cnn():

    acc_tree = 99.11

    digits = [[2], [3]]

    num_epochs = 4
    out = len(digits)
    num_run = 10

    start = time.time()

    (ds_train, ds_test) = prepare_dataset.prepare_dataset(digits)

    accuracies = []

    for i in range(num_run):
        training.train(ds_train, num_epochs, out)
        accuracies.append(testing.test(ds_test, out))

    sum_a = 0
    for a in accuracies:
        sum_a += a
    mean_a = sum_a / len(accuracies)

    print(f'Mean of the {num_run} accuracies with {num_epochs} epochs: %.2f %%' % mean_a)
    print(f'Difference is: %.2f\n' % (acc_tree - mean_a))

    finish = time.time()

    print('Seconds passed: %.3f' % (finish - start))


if __name__ == '__main__':
    mnist_cnn()
