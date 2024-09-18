from transformers import AutoTokenizer
from collections import Counter
 
# Path to the combined text file
file_path = "combined_texts.txt"
output_file = "top_30_tokens.txt"
 
# Function to count unique tokens using AutoTokenizer
def count_unique_tokens(file_path, model_name='bert-base-uncased'):
    # Load the AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Read the content of the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    # Count occurrences of each token
    token_counts = Counter(tokens)
    # Get the 30 most common tokens
    top_30_tokens = token_counts.most_common(30)
    return top_30_tokens
 
# Main function to get top tokens and write to a text file
def save_top_tokens_to_file(file_path, output_file):
    # Get the top 30 unique tokens
    top_30_tokens = count_unique_tokens(file_path)
    # Write the top 30 tokens to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Top 30 tokens and their counts:\n")
        for token, count in top_30_tokens:
            f.write(f"{token}: {count}\n")
    print(f"Top 30 tokens have been saved to {output_file}")
 
# Call the function to save tokens to the text file
save_top_tokens_to_file(file_path, output_file)