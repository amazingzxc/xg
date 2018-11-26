import os
import sys
from six.moves import xrange
import numpy
from glob import glob
from utils.data_io import get_img
class Data(object):
    def __init__(self,size,is_train=True,test_img_path=''):
        self.size=size

        if is_train:
            size_x=len(glob('./data/x_domain/*.jpg'))
            size_y=len(glob('./data/y_domain/*.jpg'))

        print('for debug use')
        use_len=12
        # use_len=min(size_x,size_y)
        self._num_examples = use_len

        if is_train:
            self.x_domain_list = glob('./data/x_domain/*.jpg')[:use_len]
            #self.x_domain_list = [if '.jpg'in tmp for tmp in os.listdir('./data/x_domain/')]
            self.y_domain_list = glob('./data/y_domain/*.jpg')[:use_len]
        else:
            self.x_domain_list = glob(test_img_path)
            self.y_domain_list = glob(test_img_path)

        self.x_domain_data = numpy.array([get_img(img,self.size) for img in self.x_domain_list])
        self.y_domain_data = numpy.array([get_img(img,self.size) for img in self.y_domain_list])

        self._index_in_epoch = 0
        self._epochs_completed = 0
        self.shape=self.x_domain_data.shape[1:]
        
    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
          fake_image = [1] * 784
          if self.one_hot:
            fake_label = [1] + [0] * 9
          else:
            fake_label = 0
          return [fake_image for _ in xrange(batch_size)], [
              fake_label for _ in xrange(batch_size)
          ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
          perm0 = numpy.arange(self._num_examples)
          numpy.random.shuffle(perm0)
          self._x_domain_data = self.x_domain_data[perm0]
          self._y_domain_data = self.y_domain_data[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Get the rest examples in this epoch
          rest_num_examples = self._num_examples - start
          x_domain_data_rest_part = self._x_domain_data[start:self._num_examples]
          y_domain_data_rest_part = self._y_domain_data[start:self._num_examples]
          # Shuffle the data
          if shuffle:
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._x_domain_data = self.x_domain_data[perm]
            self._y_domain_data = self.y_domain_data[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_num_examples
          end = self._index_in_epoch
          x_domain_data_new_part = self._x_domain_data[start:end]
          y_domain_data_new_part = self._y_domain_data[start:end]
          return numpy.concatenate((x_domain_data_rest_part, x_domain_data_new_part), axis=0) , numpy.concatenate((y_domain_data_rest_part, y_domain_data_new_part), axis=0)
        else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          return self._x_domain_data[start:end], self._y_domain_data[start:end]

if __name__=='__main__':
    data=Data()
    print(data.shape)
    test=data.next_batch(10)
    test1=data.next_batch(10)
    print(test)