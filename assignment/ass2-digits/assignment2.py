#!/usr/bin/env python
# coding: utf-8
import numpy as np
from PIL import Image
import torch as t
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# load minist dataset using sklearn with train test split (no test set)
def load_minist():
    minist_data = load_digits()
    pixel_data, labels = minist_data['data'], minist_data['target']
    return train_test_split(pixel_data, labels, test_size=0.3)

# demonstration of the dataset 
def demo():
    x, _, y, _ = load_minist()
    for idx, img in enumerate(x):
        print('*' * 10 + '%d\'s image:'%y[idx] + '*' * 10)
        plt.imshow(img.reshape(8, 8), cmap='gray')
        plt.show()
        cmd = input('Press enter to see the next image or press q to quit...\n')
        if cmd == 'q':
            break
    print('Bye~')
    return

if __name__ == '__main__':
	demo()
	# load data
	x_train, x_test, y_train, y_test = load_minist()
	print('training data size: ', x_train.shape)
	print('test data size: ', x_test.shape)

	# define your own sklearn method 
	model = None
	# to complete the codes
	# 
	#
	#
	
	# evaluate your model
	pred = model.predict(x_test)
	acc = float(np.sum((pred == y_test) / y_test.shape[0]))
	print('Your accuracy: ', acc)




