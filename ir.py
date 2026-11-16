
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample text document
text = """The students are studying Machine Learning.
They are learning Python and developing applications.
The students played games and studied different subjects."""

# Step 1: Sentence segmentation
sentences = sent_tokenize(text)
print("Sentences:")
print(sentences)

# Step 2: Tokenization - convert sentences into words
tokens = word_tokenize(text)
print("\nTokens:")
print(tokens)

# Step 3: Change case - convert all text to lowercase
text = text.lower()

# Step 4: Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))
print("\nText after removing punctuation:")
print(text)

# Step 5: Tokenize the cleaned text
tokens = word_tokenize(text)

# Step 6: Remove stop words
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word not in stop_words]

print("\nAfter stop word removal:")
print(filtered_words)

# Step 7: Stemming using Porter Stemmer
ps = PorterStemmer()
stemmed_words = [ps.stem(word) for word in filtered_words]

print("\nAfter stemming:")
print(stemmed_words)

# Step 8: POS tagging
pos_tags = nltk.pos_tag(filtered_words)

print("\nPOS Tags:")
print(pos_tags)

# Convert NLTK POS tags into WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Step 9: Lemmatization
lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word, get_wordnet_pos(tag))
    for word, tag in pos_tags
]

print("\nAfter lemmatization:")
print(lemmatized_words)

print("\nText preprocessing completed successfully!")


"=================================================================================================================="

import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK dataset
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Step 1: Create a text file
with open('file.txt', 'w', encoding='utf8') as file:
    file.write("This is the first word.\n")
    file.write("This is the second text, Hello! How are you?\n")
    file.write("This is the third, this is it now.")

# Step 2: Read the file
file = open('file.txt', 'r', encoding='utf8')
read = file.read()
file.seek(0)

print("File Content:")
print(read)

# Step 3: Count the number of lines
line = 1
for word in read:
    if word == '\n':
        line += 1

print("\nNumber of lines in file:", line)

# Step 4: Store each line in a list
array = []

for i in range(line):
    array.append(file.readline())

file.close()

print("\nLines in file:")
print(array)

# Step 5: Remove punctuation
punc = string.punctuation

for ele in punc:
    read = read.replace(ele, " ")

print("\nAfter removing punctuation:")
print(read)

# Step 6: Convert text to lowercase
read = read.lower()

print("\nAfter converting to lowercase:")
print(read)

# Step 7: Tokenize the data into individual words
def tokenize_words(file_contents):
    result = []

    for i in range(len(file_contents)):
        tokenized = []
        tokenized = word_tokenize(file_contents[i])
        result.append(tokenized)

    return result

# Apply tokenization to each line
tokenized_data = tokenize_words(array)

print("\nTokenized data:")
print(tokenized_data)

# Step 8: Remove stop words using NLTK
stop_words = set(stopwords.words('english'))

tokens_without_sw = []

for row in tokenized_data:
    filtered = [word.lower() for word in row
                if word.lower() not in stop_words]
    tokens_without_sw.append(filtered)

print("\nAfter stop word removal:")
print(tokens_without_sw)

# Step 9: Create an inverted index
# Key = word, Value = list of line numbers

dict = {}

for i in range(line):
    for item in tokens_without_sw[i]:

        if item not in dict:
            dict[item] = []

        # Avoid duplicate line numbers
        if i + 1 not in dict[item]:
            dict[item].append(i + 1)

print("\nInverted Index:")
print(dict)

# Step 10: Search for a word in the inverted index
search_word = input("\nEnter a word to search: ").lower()

if search_word in dict:
    print("Word found in line(s):", dict[search_word])

    for line_no in dict[search_word]:
        print("Line", line_no, ":", array[line_no - 1])
else:
    print("Word not found in the file.")


"=================================================================================================================="


