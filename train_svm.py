'''
Created on Aug 17, 2020

@author: michal.busta at gmail.com
'''
import numpy as np
from libsvm.svmutil import *

from sklearn.kernel_approximation import RBFSampler


if __name__ == '__main__':
  
  base_dir = '/home/busta/git/keras_OSDA'
  
  npzfile = np.load(f'{base_dir}/data_all.npz', allow_pickle=True)
  all_train_spectra = npzfile['all_train_spectra']
  all_test_spectra = npzfile['all_test_spectra']
  all_train_labels = npzfile['all_train_labels']
  all_test_labels = npzfile['all_test_labels']
  all_train_target_spectra = npzfile['all_train_target_spectra']
  all_train_target_labels = npzfile['all_train_target_labels']
  test_target_spectra = npzfile['test_target_spectra']
  test_target_labels = npzfile['test_target_labels']
  
  max_val = all_train_spectra[:, 0, :].max(0)
  x = all_train_spectra / max_val
  x = x[:, 0, :]
  
  x_test = all_test_spectra / max_val
  x_test = x_test[:, 0, :]
    
  m = svm_train(all_train_labels, x, '-s 0 -t 0 -b 1')
  
  p_label, p_acc, p_val = svm_predict(all_test_labels, x_test, m, options='-b 1')
  print(f'Test case: {x_test[0]}')
  print(f'Test label: {p_label[0]}, {p_val[0]}')
  
  #save model
  svm_save_model('test.svm', m)
  #save norm parameters
  np.savetxt('norm.csv', max_val, delimiter=',')
  np.savetxt('test_sample.csv', x_test[0:1], delimiter=',')
  
  