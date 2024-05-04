### MST AIDS 2023-2024 (Département Génie Informatique)
**Subject : The main purpose behind this lab is to get familiar with ule-based techniques, Regex, and NLP Word Embedding.**\
**Realize by : Chibani Fahd.**\
**web source : Aljarida24r.**\
**Course : NLP.**\

## 1. Introduction
In this lab, our primary objective is to acquaint ourselves with NLP's Rule-based techniques, Regex, and NLP Word Embedding. Through hands-on exploration, we aim to gain proficiency in these essential aspects of Natural Language Processing, enabling us to understand and apply them effectively in various contexts.
## 2. Upload data from MongoDB Database
In this section, we accessed our MongoDB database to upload stored data. This data was gathered by scraping content from the Aljarida24 website, 
## 3. Rule Based NLP and Regex
Imagine a scenario where a business needs a streamlined way to generate bills from raw text provided by customers. In such cases, employing Python code with Regex capabilities can be invaluable. By utilizing Regex patterns, Python scripts can effectively extract relevant information such as item names, quantities, and prices from unstructured text. This enables businesses to automate the bill generation process, saving time and reducing errors.
### 3.1 Data pre-procecing 
Our workflow commences with the essential task of text refinement, focusing on the removal of stop words and adjectives such as 'new,' 'cool','fresh'... from the bill text. This meticulous cleansing is pivotal in simplifying the subsequent regex pattern matching process. By eliminating these unnecessary elements, we enhance the text's suitability for regex pattern matching, enabling more efficient and accurate extraction of relevant information. This initial cleaning phase optimizes the text for seamless integration into regex-based algorithms
```python
Before cleaning : ['I bought three hundred two thousand twenty seven Samsung smartphones 150,333 $ each and four kilos of fresh banana for 2,4 dollar a kilogram']
After cleaning : ['bought three hundred two thousand twenty seven Samsung smartphones 150,333 $ four kilos banana 2,4 dollar kilogram']
```
### 3.2 Regex pattern :
fter meticulously cleaning our text by removing extraneous elements such as stop words and adjectives, we proceed to employ regex for pattern matching. Utilizing regex, we define a pattern with three groups: the first capturing the price, the second identifying the product name, and the third representing the unit price. This systematic approach enables us to efficiently extract relevant information from the text and generate the bill.
```python
 pattern = r"((?:" + '|'.join(numbers) + r"|\d)(?:\s(?:" + '|'.join(numbers) + r"|\d|and))*)(.*?)(\d+[\.|\,]?\d*)\b\s*(\$|dollar)"
```
To convert textual numbers into their numerical values, we create a Python script (Word2num.py File )capable of intelligently parsing numbers up to 999,999,999,999. This script efficiently take the sentence, identifies numerical representations, and transforms them into their corresponding numeric values. 

## 4.  word Embedding :
To better understand word embedding in Arabic text, we attempt to apply it to the paragraphs that we scraped in Lab 1.
### 4.1. one hot encoding :
One-hot encoding is a technique used in machine learning to represent categorical data numerically. Each category is represented as a binary vector, where only one bit is activated (set to 1) for the corresponding category, and all other bits are set to 0.
```python
Sentence : ['طالب نواب المعارضة البرلمانية']
binary representation : [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
```
### 4.2. bag of words :
The Bag of Words model represents text by counting the occurrence of words without considering grammar or order, forming a sparse numerical matrix used in various natural language processing tasks.
```python
Sentence : ['طالب نواب المعارضة البرلمانية']
Our vocabulary :{'طالب': 60, 'نواب': 72, 'المعارضة': 30, 'البرلمانية': 16, 'بالكشف': 42, 'سبل': 57, 'تيسير': 50, 'الحصول': 19, 'الدعم': 20, 'الاجتماعي': 10, 'المباشر': 29, 'ظهور': 61, 'عدد': 62, 'الاشكاليات': 14, 'ووجه': 83, 'إدريس': 5, 'السنتيسي': 21, 'رئيس': 53, 'الفريق': 27, 'الحركي': 18, 'بمجلس': 46, 'النواب': 37, 'فوزي': 64, 'لقجع': 67, 'الوزير': 39, 'المنتدب': 34, 'المكلف': 33, 'بالميزانية': 43, 'سؤالا': 55, 'كتابيا': 66, 'مفاده': 70, 'سحب': 58, 'المواطنين': 36, 'شروعهم': 59, 'الاستفادة': 12, 'وأكد': 74, 'سؤاله': 56, 'العديد': 25, 'المواطنات': 35, 'والمواطنين': 76, 'فوجئوا': 63, 'بسحب': 44, 'ومعها': 82, 'نظام': 71, 'أمو': 2, 'تضامن': 48, 'الشروع': 23, 'أشهر': 1, 'الأمر': 8, 'أثار': 0, 'امتعاض': 40, 'الفئات': 26, 'المعنية': 31, 'تقرر': 49, 'إبعادها': 4, 'البرنامج': 17, 'بمبرر': 45, 'ارتفاع': 6, 'مؤشرهم': 69, 'رغم': 54, 'أنه': 3, 'يمكن': 84, 'تتغير': 47, 'وضعيتهم': 79, 'الاجتماعية': 11, 'والاقتصادية': 75, 'الظرف': 24, 'الوجيز': 38, 'وساءل': 78, 'حقيقة': 51, 'الإجراء': 9, 'وأسبابه': 73, 'ودوافعه': 77, 'وعدد': 80, 'المعنيين': 32, 'وكذا': 81, 'انعكاساته': 41, 'الأشخاص': 7, 'قدرة': 65, 'الانخراط': 15, 'الشامل': 22, 'للأشخاص': 68, 'القادرين': 28, 'دفع': 52, 'الاشتراكات': 13}
BoW representation : [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]]
```
### 4.3. TF-IDF :
TF-IDF (Term Frequency-Inverse Document Frequency) measures word importance in a document by considering both the frequency of the term within the document and its rarity across the entire document collection, aiding in tasks like text mining and information retrieval.
```python
Sentence : ['طالب نواب المعارضة البرلمانية']
Our vocabulary :'['أثار' 'أشهر' 'أمو' 'أنه' 'إبعادها' 'إدريس' 'ارتفاع' 'الأشخاص' 'الأمر''الإجراء' 'الاجتماعي' 'الاجتماعية' 'الاستفادة' 'الاشتراكات' 'الاشكاليات''الانخراط' 'البرلمانية' 'البرنامج' 'الحركي' 'الحصول' 'الدعم' 'السنتيسي''الشامل' 'الشروع' 'الظرف' 'العديد' 'الفئات' 'الفريق' 'القادرين' 'المباشر''المعارضة' 'المعنية' 'المعنيين' 'المكلف' 'المنتدب' 'المواطنات''المواطنين' 'النواب' 'الوجيز' 'الوزير' 'امتعاض' 'انعكاساته' 'بالكشف''بالميزانية' 'بسحب' 'بمبرر' 'بمجلس' 'تتغير' 'تضامن' 'تقرر' 'تيسير''حقيقة' 'دفع' 'رئيس' 'رغم' 'سؤالا' 'سؤاله' 'سبل' 'سحب' 'شروعهم' 'طالب''ظهور' 'عدد' 'فوجئوا' 'فوزي' 'قدرة' 'كتابيا' 'لقجع' 'للأشخاص' 'مؤشرهم''مفاده' 'نظام' 'نواب' 'وأسبابه' 'وأكد' 'والاقتصادية' 'والمواطنين''ودوافعه' 'وساءل' 'وضعيتهم' 'وعدد' 'وكذا' 'ومعها' 'ووجه' 'يمكن']
TFIDF representation : [[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.5 0. 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.5 0.  0.  0.  0.  0. 0.  0.  0.  0.  0.  0.  0.  0.  0.  0. 0.  0.  0.  0.  0.  0.  0.  0. 0.  0.  0.  0.  0.  0.  0.5 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. 0.5 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]]
```
### 4.4. Word2Vec with skip Gram :
Word2Vec with skip-gram is a popular algorithm used to generate word embeddings by predicting the context words given a target word in a sentence. It learns distributed representations of words based on their co-occurrence patterns, capturing semantic similarities and relationships between words in high-dimensional vector spaces.
```python
Sentence : ['طالب نواب المعارضة البرلمانية']
Cost after epoch 4750: 1.3515463363565614
skip-grams: ['البرلمانية', 'نواب', 'طالب', 'المعارضة'] طالب
skip-grams: ['البرلمانية', 'طالب', 'نواب', 'المعارضة'] نواب
skip-grams: ['البرلمانية', 'نواب', 'طالب', 'المعارضة'] المعارضة
skip-grams: ['المعارضة', 'نواب', 'طالب', 'البرلمانية'] المعارضة
```
### 4.5. Glove and FastText :
#### 4.5.1. FastText :
FastText is a word embedding technique developed by Facebook AI Research that extends the Word2Vec model by also considering subword information. It breaks words into character n-grams and learns embeddings for these subwords, enabling it to handle out-of-vocabulary words and capture morphological similarities effectively, making it particularly useful for tasks with large vocabularies and morphologically rich languages.
```python
Word vector for : ملك
Similar words to ( ملك ) : [('الملك', 0.33138006925582886), ('للملك', 0.2525012493133545)]
```
#### 4.5.1. Glove :
GloVe (Global Vectors for Word Representation) is a word embedding model that learns vector representations of words based on their co-occurrence statistics in a corpus. It aims to capture global word co-occurrence patterns by optimizing a global objective function, producing dense word vectors that encode semantic relationships between words, which are useful for various natural language processing tasks such as word similarity and analogy detection.

## 9. What I learned
In summary, my reflections from this lab underscore the intricacies inherent in Arabic, surpassing those of its Latin counterparts. The exercise has equipped me with strategies for discerning patterns among words and discerning similarities between them. Moreover, the exploration revealed a significant research gap in Arabic NLP, resulting in a paucity of available libraries tailored to Arabic text processing, thereby posing challenges to practitioners seeking comprehensive tools in this domain.
## 10. References
<a id="1">[1]</a>Jeet. (2021, December 15). One Hot encoding of text data in Natural Language Processing. Medium. https://medium.com/analytics-vidhya/one-hot-encoding-of-text-data-in-natural-language-processing-2242fefb2148 \
<a id="2">[2]</a> Alkhatib, R. M., Zerrouki, T., Shquier, M. M. A., & Balla, A. (2023). Tashaphyne0.4: A new Arabic light stemmer based on rhizome modeling approach. Information Retrieval Journal, 26(14). doi: https://doi.org/10.1007/s10791-023-09429-y \
<a id="3">[3]</a>  notebook.community. (n.d.). https://notebook.community/arcyfelix/Courses/18-03-07-Deep%20Learning%20With%20Python%20by%20Fran%C3%A7ois%20Chollet/.ipynb_checkpoints/Chapter%206.1.1%20-%20One-hot%20encoding%20of%20words%20and%20characters-checkpoint/ \
<a id="4">[4]</a> MagedSaeed. (n.d.). GitHub - MagedSaeed/farasapy: A Python implementation of Farasa toolkit. GitHub. https://github.com/MagedSaeed/farasapy?tab=readme-ov-file#want-to-cite \
<a id="5">[5]</a> Munther, I. (2021, December 30). Sentiment Analysis of Arabic Text Data (Tweets) - Analytics Vidhya - Medium. Medium. https://medium.com/analytics-vidhya/sentiment-analysis-of-arabic-text-data-tweets-4e96c8da892b \
<a id="6">[6]</a> freeCodeCamp.org. (2019, July 24). How to process textual data using TF-IDF in Python. https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/ \
<a id="7">[7]</a>  GeeksforGeeks. (2024, January 3). Word Embedding using Word2Vec. GeeksforGeeks. https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/

