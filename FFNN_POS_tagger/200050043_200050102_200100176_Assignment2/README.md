# CS 626: Assignment 2

Part 1: Hidden Markov Model in Python using Viterbi, Vector based  
Part 2: Feed Forward Neural Network with Back Propagation in Python using Word2Vector(s)

## Team Members

- Gowri Sriya Mannepalli: 200050043
- Khyati Patel: 200050102
- Koustubh Rao: 200100176

## Installation

We used Python 3.10.6 version to run our codes. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required tools/libraries.

```bash
pip install nltk
pip install numpy
pip install matplotlib
pip install scipy
pip install -U scikit-learn
pip install --upgrade gensim
pip install torchvision
pip install joblib
```

Our versions for each of these libraries were:  
nltk: version 3.7  
numpy: version 1.23.3  
matplotlib: version 3.5.1  
scipy: version 1.9.1  
scikit-learn: version 1.1.2  
gensim: 4.2.0  
torch: 1.14.0.dev20221008  
torchaudio: 0.13.0.dev20221007  
torchvision: 0.15.0.dev20221007   
joblib: 1.2.0  

## Code files

We submitted 3 code files, one for POS tagging of input sentences, one for implementing the 3 models on brown corpus dataset and finding out various performance metrics, and one for saving the ffnn and word2vec models.

**sentence.py**:
```bash
>> python3 sentence.py
'''
Type some sentences (<=5) in terminal to get POS tags of their words according to all 3 models.
Type "Done" to break out of the loop.
'''
```

**brown_corpus.py**
```bash
>> python3 brown_corpus.py
'''
For each of 3 models (HMM-Viterbi-Symbolic, HMM-Viterbi-vector and FFNN-BP), the following things 
are printed in the terminal one by one:
 1. Scores list (accuracy) for each loop of 5-fold cross validation
 2. per POS tag Accuracy for each loop of 5-fold cross validation
 3. per POS tag Precision followed by overall precision (weighted avg)
 4. per POS tag Precision followed by overall recall (weighted avg)
 5. per POS tag F1-score followed by overall f1-score (weighted avg)
 6. per POS tag F0.5-score followed by overall f0.5-score (weighted avg)
 7. per POS tag F2-score followed by overall f2-score (weighted avg)
 8. Builds the confusion matrix using matplotlib and scikit-learn, and saves it to "img_<model_name>.png"
    in "figures" folder present in the same folder as these codes.
Finally, the time taken to run the code is printed at the bottom, before the process exits.
'''
```

**ffnn_modelsaving.py**
```bash
>> python3 ffnn_modelsaving.py
'''
Trains the FFNN-BP model on entire brown corpus as trainings dataset, and saves the trained gensim 
word vectors and trained ffnn model as "word2vec.model" and "test_model.pkl" respectively.
Seed is set to random currently, so different runs of the codes might give different values of accuracy.
'''
```

## Directory Structure

<img width="264" alt="Screenshot 2022-10-11 at 01 57 53" src="https://user-images.githubusercontent.com/87053140/194948040-68cfbdb1-5a62-4810-94e0-927604239726.png">