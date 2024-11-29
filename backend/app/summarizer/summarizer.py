# Dependencies use for TF-IDF and Transformer
import tensorflow as tf
from tensorflow import keras
from logging import PercentStyle
import math
import string
import nltk
import torch
from nltk import sent_tokenize, word_tokenize, PorterStemmer,punkt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
import gc
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# Length adjuster
def _len_convter(context, con_length):
    choice = con_length
    if (choice == 2):
        max_le = int(len(context) * 0.5)
        min_le = int(len(context) * 0.1)
    elif (choice == 1):
        max_le = int(len(context) * 0.23)
        min_le = int(len(context) * 0.06)
    elif (choice == 0):
        max_le = int(len(context) * 0.02)
        min_le = int(len(context) * 0.005)
    return max_le, min_le


# Create Frequency Table
def _create_frequency_table(text_string) -> dict:
    '''
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    Lemmatizer - an algorithm that brings words to its root word
    :return type: dict
    '''
    stopWords = set(stopwords.words("english"))
    punkt = set()
    words = word_tokenize(text_string)
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    freqTable = dict()
    for word in words:
        # word = ps.stem(word)
        word = lemmatizer.lemmatize(word)
        if (word in stopWords or word in string.punctuation):
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


# Create Frequency Matrix
def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    # ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            # word = ps.stem(word)
            word = lemmatizer.lemmatize(word)
            if (word in stopWords or word in string.punctuation):
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


# Create Document Per Words
def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


# Create TF Matrix
def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


# Create IDF Matrix
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))  # log10

        idf_matrix[sent] = idf_table

    return idf_matrix


# Create TF-IDF Matrix
def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


# Create Score Sentences
def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


# Find Average Score
def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


# Run Extractive Summariztion for Text Using TF-IDF
def run_summarization(text):
    """
    :param text: Plain summary_text of long article
    :return: summarized summary_text
    """

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 1 Sentence Tokenize
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    # print(sentences)

    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences)
    # print(freq_matrix)

    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    # print(tf_matrix)

    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    # print(count_doc_per_words)

    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    # print(idf_matrix)

    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    # print(tf_idf_matrix)

    # 7 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(tf_idf_matrix)
    # print(sentence_scores)

    # 8 Find the threshold
    threshold = _find_average_score(sentence_scores)
    # print(threshold)

    # 9 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 0.9 * threshold)
    return summary

# Model for Document Summary.
def abs_summary_for_doc(article,_len):

  _model = AutoModelForSeq2SeqLM.from_pretrained(
                "pszemraj/bigbird-pegasus-large-booksum-40k-K"
            )

  _tokenizer = AutoTokenizer.from_pretrained(
                "pszemraj/bigbird-pegasus-large-booksum-40k-K"
            )

                                           

  summarizer = pipeline(
                      "summarization", 
                      model=_model, 
                      tokenizer=_tokenizer,
                      device=0
                  )

  result = summarizer(
            article,
            min_length=_len[1], 
            max_length=_len[0],
            no_repeat_ngram_size=3, 
            encoder_no_repeat_ngram_size =3,
            clean_up_tokenization_spaces=True,
            repetition_penalty=3.7,
            num_beams=4,
            early_stopping=True
    )

  del _model
  del _tokenizer
  del summarizer
  gc.collect()
  return result

def generate_summary_for_doc(text, con_length):
    # print(tf.config.list_physical_devices('GPU'))
    # x = torch.rand(5, 3)
    # print(x)
    # print(torch.cuda.is_available())
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # print(torch.version.cuda)

    article = text.strip().replace("\n", "")
    result = run_summarization(article)
    _length = _len_convter(result, con_length)
    abs_result = abs_summary_for_doc(result, _length)

    return abs_result[0]['summary_text']


#Model for URL summary
def abs_summary_for_url(article,_len):

  tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

  model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
                                           

  summarizer = pipeline(
                      "summarization", 
                      model=model, 
                      tokenizer=tokenizer,
                      device=0
                  )

  result = summarizer(
            article,
            min_length=_len[1], 
            max_length=_len[0],
            no_repeat_ngram_size=3, 
            encoder_no_repeat_ngram_size =3,
            clean_up_tokenization_spaces=True,
            repetition_penalty=3.7,
            num_beams=4,
            early_stopping=True
    )

    
  del tokenizer
  del model
  del summarizer
  gc.collect()  
  return result

def generate_summary_for_url(text, con_length):
    # print(tf.config.list_physical_devices('GPU'))
    # x = torch.rand(5, 3)
    # print(x)
    # print(torch.cuda.is_available())
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # print(torch.version.cuda)

    article = text.strip().replace("\n", "")
    result = run_summarization(article)
    _length = _len_convter(result, con_length)
    abs_result = abs_summary_for_url(result, _length)

    return abs_result[0]['summary_text']

# Model for text
def abs_summary_for_text(article,_len):

  tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-6")

  model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-6")
                                           

  summarizer = pipeline(
                      "summarization", 
                      model=model, 
                      tokenizer=tokenizer,
                      device=0
                  )

  result = summarizer(
            article,
            min_length=_len[1], 
            max_length=_len[0],
            no_repeat_ngram_size=3, 
            encoder_no_repeat_ngram_size =3,
            clean_up_tokenization_spaces=True,
            repetition_penalty=3.7,
            num_beams=4,
            early_stopping=True
    )
    
  del tokenizer
  del model
  del summarizer
  gc.collect()
  return result

def generate_summary_for_text(text, con_length):
    # print(tf.config.list_physical_devices('GPU'))
    # x = torch.rand(5, 3)
    # print(x)
    # print(torch.cuda.is_available())
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # print(torch.version.cuda)

    article = text.strip().replace("\n", "")
    result = run_summarization(article)
    _length = _len_convter(result, con_length)
    abs_result = abs_summary_for_text(result, _length)

    return abs_result[0]['summary_text']

# Run Abstractive Summariztion For Text Using Bart

#(Code for common model for 3 pages commented below from line 256-267)

# def abs_summary(summary, _len):
#     summarizer = pipeline("summarization")
#     summarized = summarizer(summary, min_length=_len[1], max_length=_len[0])
#     return summarized

# def generate_summary(text, con_length):
#     article = text.strip().replace("\n", "")
#     result = run_summarization(article)
#     _length = _len_convter(result, con_length)
#     abs_result = abs_summary(result, _length)

#     return abs_result[0]['summary_text']


# if __name__ == '__main__':
#     # Input from user
#     article_raw = '''Machine learning and data mining often employ the same methods and overlap significantly, but while
# machine learning focuses on prediction, based on known properties learned from the training data,data mining focuses 
# on the discovery of (previously) unknown properties in the data (this is the analysis step of knowledge discovery in 
# databases). Data mining uses many machine learning methods, but with different goals; on the other hand, machine 
# learning also employs data mining methods as "unsupervised learning" or as a preprocessing step to improve learner 
# accuracy. Much of the confusion between these two research communities (which do often have separate conferences and 
# separate journals, ECML PKDD being a major exception) comes from the basic assumptions they work with: in machine 
# learning, performance is usually evaluated with respect to the ability to reproduce known knowledge, while in knowledge 
# discovery and data mining (KDD) the key task is the discovery of previously unknown knowledge. Evaluated with respect 
# toknown knowledge, an uninformed (unsupervised) method will easily be outperformed by other supervised methods, while in
# a typical KDD task, supervised methods cannot be used due to the unavailability of training data.'''
#     article = article_raw.strip().replace("\n", "")

#     # print(article)
#     # Run Extractive Summerization for given article
#     result = run_summarization(article)
#     print("The summary after tfidf : \n"+result)
#     print("The length of orginal article : "+str(len(article)))
#     print("The length after tfidf : "+str(len(result)))
#     _length = _len_convter(result)
#     abs_result = abs_summary(result, _length)
#     print("The summary after all :\n"+abs_result[0]['summary_text'])
#     print("The final summary length : "+str(len(abs_result[0]['summary_text'])))
