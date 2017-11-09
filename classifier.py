import pickle
import csv
import sys

def load_sign_names():
    # Load Sign Names File

    sign_names_file = './signnames.csv'
    sign_names_list = []
    sign_names = {}
    with open(sign_names_file) as f:
        r = csv.reader(f)
        r.__next__()
        for line in r:
            sign_names_list.append((int(line[0]), line[1]))
            sign_names[int(line[0])] = line[1]
    return sign_names
                                                            
def load_data():
    training_file = './train.p'
    validation_file = './valid.p'
    testing_file = './test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    x_train, y_train = train['features'], train['labels']
    x_valid, y_valid = valid['features'], valid['labels']
    x_test, y_test = test['features'], test['labels']

    return (x_train, y_train, x_valid, y_valid, x_test, y_test)

def print_summary(data):
    (x_train, y_train, x_valid, y_valid, x_test, y_test) = data
    # TODO: Number of training examples
    n_train = len(x_train)

    # TODO: Number of validation examples
    n_validation = len(x_valid)

    # TODO: Number of testing examples.
    n_test = len(x_test)

    # TODO: What's the shape of an traffic sign image?
    image_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

    # TODO: How many unique classes/labels there are in the dataset.

    n_classes = len(set(y_train))

    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

def main(args):
    sign_names = load_sign_names()
    data = load_data()
    print_summary(data)
    #(x_train, y_train, x_valid, y_valid, x_test, y_test) = data



if __name__ == '__main__':
    main(sys.argv[1:])
