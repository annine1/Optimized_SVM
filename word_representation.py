

import numpy as np

class WordFrequency:
  def _get_word_set(self,text): 
    sentences = []
    word_set = []
    for sent in text:
        x = [i.lower() for  i in sent.split() if i.isalpha()]
        sentences.append(x)
        for word in x:
            if word not in word_set:
                word_set.append(word)
    return set(word_set),sentences
  def _count_word(self,word,sentence):
    count = 0
    for w in sentence:
      if w==word:
        count +=1
    return count    

  def word_frequency(self,text):
    word_set , sentences = self._get_word_set(text) 
    print (sorted(word_set))
    word_count = []
    for sentence in sentences:
      row =[]
      for word in sorted(word_set):
        row.append(self._count_word(word,sentence))
      word_count.append(row)
    return np.array(word_count)

  

        


class TFIDF:

  def _get_word_set(self,text): 
    sentences = []
    word_set = []
    for sent in text:
        x = [i.lower() for  i in sent.split() if i.isalpha()]
        sentences.append(x)
        for word in x:
            if word not in word_set:
                word_set.append(word)
    return set(word_set),sentences          

  

  def _count_dict(self,sentences,word_set):
      word_count = {}
      for word in word_set:
          word_count[word] = 0
          for sent in sentences:
              if word in sent:
                  word_count[word] += 1
      return word_count
  
  #word_count = count_dict(sentences)    
  #Term Frequency
  def _termfreq(self,document, word):
      N = len(document)
      occurance = len([token for token in document if token == word])
      return occurance/N


  def _inverse_doc_freq(self,word,word_count,total_documents):
      try:
          word_occurance = word_count[word] + 1
      except:
          word_occurance = 1
      return np.log(total_documents/word_occurance)

  def tf_idf(self,text):
    word_set,sentences = self._get_word_set(text)
    #print (word_set)
    total_documents = len(sentences)
    #Creating an index for each word in our vocab.
    index_dict = {} #Dictionary to store index for each word
    i = 0
    for word in word_set:
      index_dict[word] = i
      i += 1
    #print (index_dict)  
    
    word_count = self._count_dict(sentences,word_set)
    print (len(word_count))
    vectors = []
    print (len(sentences))
    for sentence in sentences:
      tf_idf_vec = np.zeros((len(word_set),))
      for word in sentence:
          #print (word)
          tf = self._termfreq(sentence,word)
          idf = self._inverse_doc_freq(word,word_count,total_documents)
          value = tf*idf
          tf_idf_vec[index_dict[word]] = value 
      vectors.append(tf_idf_vec)    
    return np.array(vectors)   


    
    

class WordRepresentation:
  def __init__(self, method = 'tf-idf'):
    self.method = method
  def count_frequency(self,text):
      wf = WordFrequency()
      return wf.word_frequency(text)

  def tf_idf(self,text):
    return TFIDF().tf_idf(text)     
  def get_representation(self,text):
    if self.method == "tf-idf":
      return self.tf_idf(text)
      
    elif self.method=="frequency-count":
      return self.count_frequency(text)
    else:
      raise Exception("Unknown method")
    pass

