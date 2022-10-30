# Assignment 1 - First Evaluation

Hidden Markov Model in Python using Viterbi


## Team Members

- Gowri Sriya Mannepalli: 200050043
- Khyati Patel: 200050102
- Koustubh Rao: 200100176


## Installation

We used Python 3.8.10 version to run our codes. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required tools/libraries.

```bash
  pip install nltk
  pip install numpy
  pip install matplotlib
  pip install scipy
  pip install -U scikit-learn
```

Our versions for each of these libraries were:  
NLTK: version 3.7  
Numpy: version 1.22.2  
Matplotlib: version 3.5.3  
Scipy: version 1.9.1  
Scikit-learn:  version 0.0


## Code files

We submitted 2 code files, one for POS tagging of input sentences, and one for implementing HMM using Viterbi on brown corpus dataset and finding out various performance metrics, etc.  
  
**sentence.py**:  
```bash
>> python3 sentence.py
# Type some sentences (<=5) in terminal to get POS tags of their words  
# Type 'Done' to break out of the loop  
# At the end, runtime (in seconds) is printed in the terminal
```

**brown_corpus.py**
```bash
>> python3 brown_corpus.py
# Took ~180 seconds to run in our systems  
# 1. Prints scores list (accuracy) for each loop of 5-fold cross validation  
# 2. Prints per POS tag Accuracy for each loop of 5-fold cross validation  
# 3. Prints per POS tag Precision followed by overall precision (weighted avg)  
# 4. Prints per POS tag Precision followed by overall recall (weighted avg)  
# 5. Prints per POS tag F1-score followed by overall f1-score (weighted avg)  
# 6. Prints per POS tag F0.5-score followed by overall f0.5-score (weighted avg)  
# 7. Prints per POS tag F2-score followed by overall f2-score (weighted avg)  
# 8. Builds the confusion matrix using matplotlib, and saves it to "img.png" in same folder as the codes  
# At the end, runtime (in seconds) is printed in the terminal
```