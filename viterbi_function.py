import nltk
import numpy as np
from viterbi import viterbi

# load the Brown corpus tagged sentences (limited to the first 10,000)
corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]

# get universal POS tags
universal_tags = list(set(tag for sent in corpus for _, tag in sent))
num_states = len(universal_tags)

# create a dictionary to map POS tags to integers
tag_to_int = {tag: i for i, tag in enumerate(universal_tags)}

# initialize a dictionary to store the frequency of each tag as the first word in a sentence
initial_tag_counts = {tag: 0 for tag in universal_tags}
total_sentences = len(corpus)

# calculate the frequency of the first word's POS tag (for initial probabilities)
for sentence in corpus:
    if sentence:  # Ignore empty sentences
        first_word_tag = sentence[0][1]  # POS tag of the first word
        initial_tag_counts[first_word_tag] += 1

# initialize and calculate initial probabilities (pi)
pi = np.zeros(num_states, dtype=np.float_)
for tag, count in initial_tag_counts.items():
    pi[tag_to_int[tag]] = count / total_sentences

# initialize and calculate transition probabilities (A)
transition_counts = {}
A = np.zeros((num_states, num_states), dtype=np.float_)

for sentence in corpus:
    obs_ints = [tag_to_int.get(tag) for word, tag in sentence]
    for i in range(len(obs_ints) - 1):
        current_int = obs_ints[i]
        next_int = obs_ints[i + 1]

        if current_int in transition_counts:
            if next_int in transition_counts[current_int]:
                transition_counts[current_int][next_int] += 1
            else:
                transition_counts[current_int][next_int] = 1
        else:
            transition_counts[current_int] = {next_int: 1}

        total_count = sum(transition_counts[current_int].values())
        A[current_int, next_int] = (
            transition_counts[current_int][next_int] / total_count
        )

# create a list of all words in the corpus + add OOV function (append("UNK"))
all_words = list(set(word for sentence in corpus for word, _ in sentence))
all_words.append("UNK")

# create a dictionary to map words to their indices
word_to_index = {word: i for i, word in enumerate(all_words)}
num_words = len(all_words)

# initialize and calculate observation probabilities (B)
B = np.ones((num_states, num_words), dtype=np.float_)  # add smoothing function
for sentence in corpus:
    for word, tag in sentence:
        state_index = universal_tags.index(tag)
        if word in word_to_index:
            word_index = word_to_index[word]
        else:
            word_index = word_to_index[
                "UNK"
            ]  # Use the index for "UNK" if the word is unknown
        B[state_index][word_index] += 1

# normalize observation probabilities to probability values
B /= np.sum(B, axis=1, keepdims=True)

# load test sentences from the Brown corpus (sentences 10150 to 10152)
test_corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]
results = []

# run Viterbi algorithm for each test sentence
for i in range(len(test_corpus)):
    test_sentence = test_corpus[i]
    obs = []

    for word in test_sentence:
        if word[0] in word_to_index:
            obs.append(word_to_index[word[0]])
        else:
            obs.append(word_to_index["UNK"])
    output = viterbi(obs, pi, A, B)
    results.append(output)

# print the results
total_tests = len(results)
total_correct = 0
total_incorrect = 0

for i, result in enumerate(results):
    tag_sequences = [tag_to_int[tag] for word, tag in test_corpus[i]]
    viterbi_result = result

    first_array = viterbi_result[0]
    correct = 0
    incorrect = 0
    incorrect_words = []  # List to store mismatched words and tags

    for j in range(len(tag_sequences)):
        if tag_sequences[j] == first_array[j]:
            correct += 1
        else:
            incorrect += 1
            incorrect_words.append((test_corpus[i][j][0], test_corpus[i][j][1]))

    total_correct += correct
    total_incorrect += incorrect

    print(f"< {10150+i} of Brown Corpus >")
    print(f"Test Result: {viterbi_result}")
    print(
        f"Expected Integer Sequence: {tag_sequences}"
    )  # Print expected integer sequence
    print(f"Correct elements: {correct}")
    print(f"Incorrect elements: {incorrect}")
    print(f"Mismatched words and tags: {incorrect_words}")
    print()

accuracy = total_correct / (total_correct + total_incorrect) * 100  # Calculate accuracy
print(f"\nOverall accuracy(%): {accuracy:.3f} %")  # Print overall accuracy
