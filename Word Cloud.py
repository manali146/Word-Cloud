# Python program to generate WordCloud

# importing all necessary modules
import os
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import textract
import string
from collections import Counter

# STEP 1: READ THE FILE

def read_file_textract(filepath):
    text = textract.process(filepath)
    return text.decode("utf-8") 

def read_file(filepath, use_method = 'textract'):
    text = ""
    if not os.path.isfile(filepath):
        print(f'Invalid file:{filepath}')
    else:
        if use_method == 'textract':
            return read_file_textract(filepath)
        else:
            print('Invalid method to read file. Supported formats: "textract"')
    
    return text

# STEP 2: IDENTIFY THE KEYWORDS

def extract_keywords(text, ignore_words = [],
                     min_word_length = 0,
                     ignore_numbers = True,
                     ignore_case = True):
    # Remove words with special characters
    filtered_text = ''.join(filter(lambda x:x in string.printable, text))
    
    # Create word tokens from the text string
    tokens = word_tokenize(filtered_text)
    
    # List of punctuations to be ignored 
    punctuations = ['(',')',';',':','[',']',',','.','--','-','#','!','*','"','%']
    
    # Get the stopwords list to be ignored
    stop_words = stopwords.words('english')

    # Convert ignore words from user to lower case
    ignore_words_lower = [x.lower() for x in ignore_words]
    
    # Combine all the words to be ignored
    all_ignored_words = punctuations + stop_words + ignore_words_lower
    
    # Get the keywords list
    keywords = [word for word in tokens \
                    if  word.lower() not in all_ignored_words
                    and len(word) >= min_word_length]    

    # Remove keywords with only digits
    if ignore_numbers:
        keywords = [keyword for keyword in keywords if not keyword.isdigit()]

    # Return all keywords in lower case if case is not of significance
    if ignore_case:
        keywords = [keyword.lower() for keyword in keywords]
    
    return keywords

# STEP 3: CREATE THE WORD CLOUD

def create_word_cloud(keywords, maximum_words = 150, bg = 'white', cmap='Dark2',
                     maximum_font_size = 256, width = 800, height = 800, 
                     random_state = 42, fig_w = 15, fig_h = 10, output_filepath = "Clouds/4"):
    
    # Convert keywords to dictionary with values and its occurences
    word_could_dict=Counter(keywords)

    #mask = np.array(Image.open("city.jpg"))
    wordcloud = WordCloud(background_color=bg, max_words=maximum_words, colormap=cmap, 
                          stopwords=STOPWORDS, max_font_size=maximum_font_size,
                          random_state=random_state, 
                          width=width, height=height,contour_width=3).generate_from_frequencies(word_could_dict)
    
    plt.figure(figsize=(fig_w,fig_h))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if output_filepath:
        plt.savefig(output_filepath, bbox_inches='tight')
    plt.show()
    plt.close()

""" filepath = "AReviewof311inNewYorkCity.pdf"
file_text = read_file(filepath)
keywords = extract_keywords(file_text, min_word_length = 3)
create_word_cloud(keywords,maximum_words=200) """

# WORDCLOUD FOR ALL FILES IN A FOLDER


docs_path = '000Related Work'
ignore_words = ['Fig','like','e.g.','i.e.','one','data','also']
all_keywords = []

for filename in os.listdir(docs_path):
    filepath = os.path.join(docs_path, filename)
    if os.path.isfile(filepath) and filename.endswith('.pdf'):
        print(f'Parsing file: {filename}')
        try:
            file_text = read_file(filepath)
            keywords = extract_keywords(file_text,min_word_length = 3, ignore_words = ignore_words)
            all_keywords.extend(keywords)
        except:
            print(f'ERROR!!! Unable to parse file: {filename}. Ignoring file!!')
        

print(f'Completed reading all pdf files in folder:{docs_path}')

create_word_cloud(all_keywords, bg = 'black', cmap = 'Set2',random_state = 100, width = 800, height = 800)

# ADDON: LIST OF TOP KEYWORDS AS DATAFRAME

pd.set_option("max_rows", None)
distinct_keywords_df = pd.DataFrame(all_keywords,columns=['keywords']).value_counts().rename_axis('keyword').reset_index(name='count')
distinct_keywords_df['word_len'] = distinct_keywords_df['keyword'].apply(lambda x: len(x))
distinct_keywords_df.head(10)