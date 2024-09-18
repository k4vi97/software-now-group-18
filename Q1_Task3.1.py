import os
import csv
from collections import Counter
 
# Path to the combined text file
file_path = "combined_texts.txt"
 
# Function to count word occurrences
def count_word_occurrences(file_path):
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Convert to lowercase for case-insensitive counting
    # Tokenize the text into words
    words = text.split()
    # Count occurrences of each word
    word_counts = Counter(words)
    # Get the 30 most common words
    top_30_words = word_counts.most_common(30)
    return top_30_words
 
# Get the top 30 words
top_30_words = count_word_occurrences(file_path)
 
# Write the top 30 words and their counts into a CSV file
output_csv = "top_30_words.csv"
with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Word', 'Count'])  # Header
    writer.writerows(top_30_words)
 
print(f"Success: Top 30 words have been stored in {output_csv}")