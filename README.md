

# Alex-SVM

Example usage of libsvm in javascript with (https://github.com/mljs/libsvm) 

## Requirements: 

```
npm install libsvm-js
```

## Fix the memory issue

```
cd node_modules/libsvm-js/out/asm/ 
sed -i 's/16777216/26777216/g' libsvm.js
```

## Usage

Train a model: train_svm.py: modify base_dir = '/home/busta/git/keras_OSDA' to directory with saved data (expected file: data_all.npz)

```
python3 train_svm.py 
```

Test code in js:

```
node --inspect hello-world-svm.js
```

## TODO 
 - grid search for best parameters of C-SVC 
 