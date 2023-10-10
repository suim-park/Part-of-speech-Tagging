# HW 3. POS Tagging
* __`Class`__: Introduction to Natural Language Processing
* __`Professor`__: Patrick Wang
* __`Name`__: Suim Park (sp699)
## Part-of-speech-Tagging
Generate the components of a part-ofspeech hidden markov model: the transition matrix, observation matrix, and initial state distribution.
## Question
Use the first 10k tagged sentences from the Brown corpus to generate the components of a part-ofspeech hidden markov model: __the transition matrix, observation matrix, and initial state distribution__. Use the universal tagset:
```Python
nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
```
Also hang on to the mappings between states/observations and indices. Include an `OOV/UNK observation` and `smoothing` everywhere. Using the provided Viterbi implementation, infer the sequence of states for sentences 10150-10152 of the Brown corpus:
```Python
nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
```
and compare against the truth. __Explain why your POS tagger does or does not produce the correct tags__. You may work in a group of 1 or 2. Submissions will be graded without regard for the group size. You should turn in a document (.txt, .md, or .pdf) answering all of the red items above. You should also turn in Python scripts (.py) for each of the blue items. Unless otherwise specified, you may use only numpy and the standard library.

## Results
```Python
< 10150 of Brown Corpus >
Test Result: ([9, 6, 2, 5, 6, 11, 11, 9, 6, 8, 11, 11, 10], 6.975978723562705e-45)
Expected Integer Sequence: [9, 11, 2, 5, 6, 11, 11, 9, 6, 8, 11, 11, 10]
Correct elements: 12
Incorrect elements: 1
Mismatched words and tags: [('coming', 'VERB')]

< 10151 of Brown Corpus >
Test Result: ([9, 5, 6, 11, 9, 5, 6, 6, 2, 9, 5, 6, 11, 11, 2, 4, 6, 10], 1.0399079756652066e-61)
Expected Integer Sequence: [9, 5, 6, 11, 9, 5, 5, 6, 2, 9, 5, 6, 11, 11, 2, 4, 9, 10]
Correct elements: 16
Incorrect elements: 2
Mismatched words and tags: [('face-to-face', 'ADJ'), ('another', 'DET')]

< 10152 of Brown Corpus >
Test Result: ([7, 11, 9, 5, 6, 2, 9, 6, 2, 9, 5, 6, 3, 9, 6, 10], 5.098984326459768e-44)
Expected Integer Sequence: [7, 11, 9, 5, 6, 2, 9, 6, 2, 9, 5, 6, 3, 9, 6, 10]
Correct elements: 16
Incorrect elements: 0
Mismatched words and tags: []


Overall accuracy(%): 93.617 %
```

```Python
# Print transition probabilities for error checking (if you want to see, add this code at the end of the code)
for i, from_tag in enumerate(universal_tags):
    print(f"Transition Probabilities from '{from_tag}':")
    for j, to_tag in enumerate(universal_tags):
        print(f"  '{to_tag}': {A[i, j]}")
```

## Explain the results
Hidden Markov Model (HMM) based Part-of-Speech (POS) tagging can be summarized as follows:
```
In the first sentence, ('coming', 'VERB') / Test result 'NOUN':
Transition probability:
DET -> VERB: 0.0615
DET -> NOUN: 0.6156 (error)
```
- The actual model expected 'coming(VERB)' to follow 'The(DET)' in the sentence. However, the test result indicated 'NOUN.' This discrepancy arises because the transition probability from DET to VERB is 0.0615, while the transition probability from DET to NOUN is 0.6156, which is significantly higher. With such a substantial difference in transition probabilities and relatively small observation probabilities, errors like this can occur.
```
In the second sentence, ('face-to-face', 'ADJ') / Test result 'NOUN':
Transition probability:
ADJ -> ADJ: 0.0592
ADJ -> NOUN: 0.6749 (error)
```
- The actual model expected 'face-to-face(ADJ)' to follow 'introductory(ADJ)' in the sentence. However, the test result indicated 'NOUN.' This error also occurs due to the significant difference between the transition probability from ADJ to ADJ and the transition probability from ADJ to NOUN. Considering the grammatical structure where adjectives are followed by nouns in sentences, the difference in probabilities can be better understood as the reason for this error.
```
In the second sentence, ('another', 'DET') / Test result 'NOUN':
Transition probability:
NUM -> DET: 0.0104
NUM -> NOUN: 0.3827 (error)
```
- The actual model expected 'another(DET)' in the sentence, but the test result indicated 'NOUN.' This error is also caused by the remarkably high probability of a noun following a number. Considering the grammar where numbers precede nouns, we can understand why such a difference in transition probabilities led to this error.</br>
## Summary
HMM models can produce various types of errors. In this test, there are also out-of-vocabulary (OOV) words that were treated as 'UNK (Unknown)' and correctly tagged by the model, such as `[('denominations', 'NOUN')]` in the first sentence and `[('preparatory', 'ADJ'), ('introductory', 'ADJ'), ('face-to-face', 'ADJ')]` in the second sentence. The key issue lies in the tagging process of words, where significant differences in transition probabilities play a crucial role in causing errors.
