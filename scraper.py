import requests
from bs4 import BeautifulSoup
import re
from functools import reduce
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

class scraper:
  def main():
    self.url = "https://quizlet.com/2832581/barrons-333-high-frequency-words-flash-cards/"
    
  def scrape_words(self, url=0)
    if (url):
      URL = url
    else:
      URL = self.url
    r = requests.get(URL) # request object collects server's response to the http request.
    soup = BeautifulSoup(r.content, 'html5lib') # BeautifulSoup creates a parser tree out of the html response that was collected using request.
    rows = soup.find_all('div', class_='SetPageTerm-inner') # Looking for elements with tag='div' and class_='SetPageTerm-inner'.
    dic = {}
    for row in rows: # iterating over all the elements.
        part = row.find_all('span', class_='TermText notranslate lang-en') # Looking for elements with tag='span' and class_='TermText notranslate lang-en'
        word = part[0].text # collecting the words.
        meaning = part[1].text # collecting the meaning.
        dic[word] = meaning # adding the word, meaning to dictionary as key value pairs.
    df = pd.DataFrame(dic.items(), columns=['word', 'meaning'], index='False') # converting to dataframe
    return df
    
  def synonyms(self, word, th=20):
    URL = f"https://www.thesaurus.com/browse/{word}"
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, 'html5lib')
    rows = soup.find_all('span', class_ = 'css-133coio etbu2a32')
    syn = [word]
    if(len(rows)<th):
      th = len(rows)
    for r in rows[:th]:
      try:
        syn.append(r.a.text)
      except:
        syn.append(r.span.text)
    return syn
    
  def scrape_synonyms(self):
    df = scrape_words()
    mapping = [] # stores the word and it's synonyms
    for word in tqdm_notebook(df.word.values): # iterating over the words and scrapping synonyms
      syn = synonyms(word, th=5) # I'll be collecting 5 synonyms
      mapping.append(syn) # storing the synonyms.
    data = pd.DataFrame(mapping) # Converting to dataframe
    data.columns = ['word','synonym_1','synonym_2','synonym_3','synonym_4','synonym_5']
    return data
    
  def scrape_data(self):
    df = scrape_synonyms()
    data = scrape_words()
    result = pd.merge(df, data, on='word')
    result.fillna('', inplace=True)
    return result
