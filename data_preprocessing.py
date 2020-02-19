import regex as re
class preprocessing():
  def main():
    self.stop_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there",
              "about", "once", "during", "out", "very", "having", "with", "they",
              "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", 
              "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", 
              "as", "from", "him", "each", "the", "themselves", "until", "below", 
              "are", "we", "these", "your", "his", "through", "don", "nor", "me", 
              "were", "her", "more", "himself", "this", "down", "should", "our", 
              "their", "while", "above", "both", "up", "to", "ours", "had", "she",
              "all", "no", "when", "at", "any", "before", "them", "same", "and",
              "been", "have", "in", "will", "on", "does", "yourselves", "then", 
              "that", "because", "what", "over", "why", "so", "can", "did", "not",
              "now", "under", "he", "you", "herself", "has", "just", "where", "too",
              "only", "myself", "which", "those", "i", "after", "few", "whom", "t", 
              "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", 
              "how", "further", "was", "here", "than"]

  def preprocess(self, sentence):

    sentence_clean = [words if words not in self.stop_words else '' for words in sentence.split(' ')]
    sentence = ' '.join(sentence_clean)

    sentence = re.sub(';', '', sentence)
    sentence = re.sub('\(', '', sentence)
    sentence = re.sub('\)', '', sentence)
    sentence = re.sub(',', '', sentence)
    sentence = re.sub('-', ' ', sentence)
    sentence = re.sub('\d', '', sentence)
    sentence = re.sub('  ', ' ', sentence)

    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)

    return sentence
    
