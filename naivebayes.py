import math
import re

class Bayes_Classifier:

    def __init__(self):
        self.class_word_counts = {}
        self.class_counts = {}
        self.vocab = set()
        #from NLTK Corpus
        self.stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]



    def train(self, lines):
        for line in lines:
            label, ID, review = line.split('|')
            label = int(label.strip())
            review = review.lower()
            words1 = re.findall(r'\b\w+\b', review)
            words = []
            for i in words1:
                if i not in self.stopwords:
                    words.append(i)
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
            if label not in self.class_word_counts:
                self.class_word_counts[label] = {}
            for word in words:
                if word not in self.class_word_counts[label]:
                    self.class_word_counts[label][word] = 0
                self.class_word_counts[label][word] += 1
                self.vocab.add(word)


    def classify(self, lines):
        results = []
        for line in lines:
            label, ID, review = line.split('|')
            review = review.lower()
            words1 = re.findall(r'\b\w+\b', review)
            words = []
            for i in words1:
                if i not in self.stopwords:
                    words.append(i)
            posterior = {}
            V = len(self.vocab)
            for label in self.class_counts.keys():
                prior = math.log(self.class_counts[label] / sum(self.class_counts.values()))
                likelihood = 0
                for word in words:
                    #add-one smoothing
                    word_count = self.class_word_counts[label].get(word, 0) + 1
                    total_words = V + sum(self.class_word_counts[label].values())
                    likelihood += math.log(word_count / total_words)
                posterior[label] = prior + likelihood
            results.append(str(max(posterior, key=posterior.get)))
        return results
