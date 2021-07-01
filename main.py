import testing
import training
import time
import prepare_dataset
import variance


def mnist_cnn():

    acc_tree_a = 98.63
    acc_tree_b = 98.53

    digits = [[0], [3], [6], [7]]

    num_epochs = 2
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

    var = variance.variance(accuracies)

    print(f'Mean of the {num_run} accuracies with {num_epochs} epochs: %.2f %%' % mean_a)
    print(f'Variance: {var}\n')
    print(f'Difference is: %.2f' % (acc_tree_a - mean_a))
    print(f'Difference is: %.2f\n' % (acc_tree_b - mean_a))

    finish = time.time()

    print('Seconds passed: %.3f' % (finish - start))


if __name__ == '__main__':
    mnist_cnn()
