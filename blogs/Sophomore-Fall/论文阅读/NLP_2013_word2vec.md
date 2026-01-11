## Core understanding

- **problem definition**: we need to find a way( a new training architecture) to train quicker and perform better even we don't have too much data
- **current solutions**: use NNLM to do everything

RNNLM: use recurrent NNLM: can't  be expanded 

![image-20251016183712975](D:\wesleyfei1.github.io\blogs\Sophomore-Fall\论文阅读\NLP_2013_word2vec.assets\image-20251016183712975.png)

- **core architecture**:

**cbow && skip-gram architecture**:

after embedding, directly use a "window" to distribute/conclude the information from the context, and than project it to use softmax to choose the best target

**key architecture& the model**:

![image-20251016183801433](D:\wesleyfei1.github.io\blogs\Sophomore-Fall\论文阅读\NLP_2013_word2vec.assets\image-20251016183801433.png)

![image-20251017212145631](D:\wesleyfei1.github.io\blogs\Sophomore-Fall\论文阅读\NLP_2013_word2vec.assets\image-20251017212145631.png)

remove the hidden layer, use a **sum-up method** to sum take average of the up&down context to get a overall understanding of the solutions

- experiment:

**task:** syntactic questions(语法关系)， and semantic questions

v(B)-v(A)+v(C)=?v(D)

**result**: skip-gram$\approx$ cbow>nnlm>rnnlm



larger vector-dimensionality, larger training words, more epoch the better

skip-gram>cbow, and larger vector dimensionality && training words need to increase in corporation to ++ accuracy



- **some useful ideas**: Huffman coding to quicken our softmax,  nagative sampling

the ideas of concluding message of up/down context simply to train the network（transformer&&

provide a perspective to train the word2vec(use king-man+woman=queen), to construct a test set



i think this is  is very similar to that in transformer



