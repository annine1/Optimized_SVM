from nltk.stem import PorterStemmer
import re
class DataPreprocessing:
  
  def _remove_html(self, data):
    for i in range(len(data)) :
      p = re.compile(r'<.*?>')
      data[i] = p.sub('', data[i])
      
    return data

  #lowercase
  def _convert_to_lower_case(self,data):
    for i in range(len(data)):
      data[i] = data[i].lower()
    return data  

  #single characters
  def _remove_single_characters(self, data):
    for i in range(len(data)) :  
      data[i] = " ".join([w for w in data[i].split() if len(w)>1]) 
    return data 

   #punctuation  
  def _remove_punctuation(self,data):
    symbols = "!,\"#$%&()*+-./:;<=>?@[\]^_`{|}~'\n"
    for symbol in symbols:
      for i in range(len(data)):
        data[i] = data[i].replace(symbol,"")
    return data  

  #stemming  
  def _stem(self,data):
    ps = PorterStemmer ()
    
    for i in range(len(data)) :
      new_doc = ""
      for word in data[i].split():
        new_doc+= ps.stem(word)+" "
      data[i] = new_doc  
    return data   
  
  def preprocess(self,data):
     data = self._remove_html(data)
     data = self._remove_single_characters(data)
     data = self._convert_to_lower_case(data)
     data = self._remove_punctuation(data)
     data = self._stem(data)

     return data