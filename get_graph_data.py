import csv
import string

positive_count = 0
negative_count = 0
neutral_count = 0


score_counts = {1:0, 2:0, 3:0, 4:0, 5:0}

word_dictionary_count = {}


# Read CSV and replace any non-utf-8 charactes with "?"
with open ('csv_files/Equal_unmodified.csv','r', errors='replace') as csv_file:
    reader = csv.reader(csv_file)
    next(reader) # Skip the row that contains the column names.
    for row in reader:
        
        product_name = row[0]
        product_price = row[1]
        product_rating_number = row[2]
        product_review_short_message = row[3]
        product_review_short_summary = row[4]
        product_sentiment = row[5]

        combined_text = product_review_short_message + " " + product_review_short_summary
        combined_text = combined_text.translate(str.maketrans('', '', string.punctuation)).lower()
        
        if(len(product_rating_number) > 1):
            # invalid data
            continue


        

        if product_sentiment == "positive":
            positive_count += 1
        elif product_sentiment == "negative":
            negative_count += 1
        elif product_sentiment == "neutral":
            neutral_count += 1
        score_counts[int(product_rating_number)] += 1


        # Count words for word cloud.
        for word in combined_text.split(" "):
            if word in word_dictionary_count:
                word_dictionary_count[word] += 1
            else:
                word_dictionary_count[word] = 1





print("Positive count: " + str(positive_count))
print("Negative count: " + str(negative_count))
print("Neutral count: " + str(neutral_count))

print("Score Counts: " + str(score_counts))


print(word_dictionary_count)




