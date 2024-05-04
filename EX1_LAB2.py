import Word2num as w2n
from tabulate import tabulate
import re
from nltk.corpus import stopwords
import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

text = "I bought three hundred two thousand twenty seven Samsung smartphones 150,333 $ each, acquired five smartphones for 145$ each four kilos of fresh banana for 2,4 dollar a kilogram and one Hamburger with 4,5 dollar, ten boxes of tisseues for 2.5 $ each "

# Tokenize the Arabic text
word_tokens = nlp(text)

# Filter out tokens that are not stop words or adjectives
filtered_tokens = [token.text for token in word_tokens if token.pos_ != "ADJ"]

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Filter out stop words
filtered_tokens = [token for token in filtered_tokens if token.lower() not in stop_words]
filtered_tokens = ' '.join(filtered_tokens)
print(filtered_tokens)
# Define regex pattern to match weight units and words ending with "gram" or "grams"
weight_units_regex = r'\b(?:kg|kilos|lb|pound|[\w]*(?:gram|grams))\b'
filtered_tokens = re.sub(weight_units_regex, '', filtered_tokens)

numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
           'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
           'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million']
pattern = r"((?:" + '|'.join(numbers) + r"|\d)(?:\s(?:" + '|'.join(
    numbers) + r"|\d|and))*)(.*?)(\d+[\.|\,]?\d*)\b\s*(\$|dollar)"

matches = re.finditer(pattern, filtered_tokens)

# Initialize lists to store data for each column
products = []
quantities = []
unit_prices = []
total_prices = []

# Process each match
for match in matches:
    quantity = match.group(1)
    # Remove commas from the quantity string and convert it to a number
    quantity = w2n.word_to_num(quantity.replace(',', '.'))
    product = match.group(2)
    price = float(match.group(3).replace(',', '.'))  # Remove commas from the price string

    # Calculate total price
    total_price = quantity * price

    products.append(product)
    quantities.append(quantity)
    unit_prices.append(price)
    total_prices.append(total_price)

# Format data into a list of lists for tabulate
bill_data = list(zip(products, quantities, unit_prices, total_prices))

# Print the bill as a table
print(tabulate(bill_data, headers=["Product", "Quantity", "Unit Price ($)", "Total Price ($)"], tablefmt="pretty"))