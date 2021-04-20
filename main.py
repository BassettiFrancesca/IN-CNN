import testing
import training
import time
import prepare_dataset


def mnist_cnn():

    acc_tree_a = 97.10
    acc_tree_b = 95.60

    digits = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

    num_epochs = 4
    out = len(digits)
    num_run = 3

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
    print(f'Difference is: %.2f' % (acc_tree_a - mean_a))
    print(f'Difference is: %.2f\n' % (acc_tree_b - mean_a))

    finish = time.time()

    print('Seconds passed: %.3f' % (finish - start))


if __name__ == '__main__':
    mnist_cnn()
