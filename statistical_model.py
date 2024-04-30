import collections
import math
import sys
import numpy as np

class LanguageModel:
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"
    n_gram_sub_counts = None
    n_gram_counts = None
    tokens = None
    vocab = None
    n_grams = None
    token_to_n_grams = {}

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
        Parameters:
            n_gram (int): the n-gram order of the language model to create
            is_laplace_smoothing (bool): whether or not to use Laplace smoothing
        """
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing

    def make_n_grams(self, tokens, size):
        """
        returns list of tokens in doc
        :param doc: document to tokenize
        :return (list): list of n_grams
        """
        n_grams = []
        if size == 0:
            return []
        for i in range(len(tokens) - size + 1):
            n_grams.append(tuple(tokens[i:i + size]))
        return n_grams

    def replace_unk(self, tokens):
        """
            Returns list of tokens with unknown words replaced
        :param tokens: unfiltered tokens
        :return: updated tokens
        """
        counts = collections.Counter(tokens.copy())
        for i, n_gram in enumerate(tokens):
            if counts[n_gram] == 1:  # maybe have to check if is sentence token
                tokens[i] = self.UNK
        return tokens

    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Parameters:
          training_file_path (str): the location of the training data to read

        Returns:
        None
        """
        f = open(training_file_path, "r")
        doc = f.read().split()
        f.close()

        self.tokens = self.replace_unk(doc)
        self.vocab = set(self.tokens)
        n_grams_sub = self.make_n_grams(self.tokens, self.n_gram - 1)
        n_grams = self.make_n_grams(self.tokens, self.n_gram)
        self.n_gram_sub_counts = collections.Counter(n_grams_sub)
        self.n_gram_counts = collections.Counter(n_grams)
        self.n_grams = set(n_grams)

        for n_gram in n_grams:
            token = n_gram[:-1]
            if token not in self.token_to_n_grams:
                self.token_to_n_grams[token] = set()
            self.token_to_n_grams[token].add(n_gram)

    def n_gram_probability(self, n_gram, use_smoothing, tokens):
        """
            Gets the probability of the given n_gram
        :param n_gram: n_gram to calculate probability for
        :param use_smoothing: if laplace smoothing should be used
        :param tokens: tokens to calculate probability with
        :return: probability
        """
        numerator = self.n_gram_counts[n_gram]
        denominator = self.n_gram_sub_counts[n_gram[:-1]] if self.n_gram > 1 else len(tokens)
        if use_smoothing:
            numerator += 1
            denominator += len(self.vocab)
        return 0 if denominator == 0 else numerator / denominator

    def replace_unk_tokens(self, n_gram):
        """
            Replaces unknown tokens in n_gram with UNK
        :param n_gram: given n_gram to replace unknown tokens
        :return: updated n_gram with unknown tokens
        """
        tup = tuple()
        for word in n_gram:
            if word not in self.vocab:
                tup += (self.UNK,)
            else:
                tup += (word,)
        return tup

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of

        Returns:
          float: the probability value of the given string for this model
        """
        n_grams = self.make_n_grams(sentence.split(), self.n_gram)
        log_sum = 0
        for n_gram in n_grams:
            n_gram = self.replace_unk_tokens(n_gram)
            prob = self.n_gram_probability(n_gram, self.is_laplace_smoothing, self.tokens)
            if prob == 0:
                return 0
            log_sum += math.log(prob)
        return math.pow(math.e, log_sum)

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          str: the generated sentence
        """
        return self.generate_multigram_sentences()

    def generate_multigram_sentences(self):
        """
            generates a single sentence for multigram model
        :return: generated sentence
        """
        last_word = tuple([self.SENT_BEGIN for i in range(self.n_gram - 1)])
        sentence = list(last_word)
        while last_word[-1] != self.SENT_END:
            n_grams = self.token_to_n_grams[last_word]
            n_gram_probs = [self.n_gram_probability(n_gram, False, self.tokens) for n_gram in n_grams]
            i = np.random.choice(len(n_grams), p=n_gram_probs)
            last_word = list(n_grams)[i][1:]
            sentence.append(last_word[-1])
        for i in range(self.n_gram - 2):
            sentence.append(self.SENT_END)

        return " ".join(sentence)

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
          n (int): the number of sentences to generate
        Returns:
          list: a list containing strings, one per generated sentence
        """
        generated_sentences = []
        while len(generated_sentences) < n:
            sentence = self.generate_sentence()
            # only keep sentences which are <= 140 characters + 27 characters for start and end tokens
            if len(sentence) < (140 + 27):
                generated_sentences.append(sentence)
        return generated_sentences

def test_model(model, test_file, model_name):
    print("Model:", model_name)
    generated_sentences = model.generate(20)
    print("Sentences:")
    print("\n".join(generated_sentences), "\n")
    f = open(test_file, "r")
    sentences = f.read().split("\n")
    f.close()
    scores = []
    for sentence in sentences:
        if len(sentence) != 0:
            scores.append(model.score(sentence))
    average = np.average(scores)
    std_dev = np.nanstd(scores)
    print("test corpus: ", test_file)
    print("# of test sentences:", len(scores))
    print("Average probability:", average)
    print("Standard deviation:", std_dev)
