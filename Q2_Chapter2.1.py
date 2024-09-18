### Chapter 2.1: The Chamber of Strings
 
# Input string containing a mix of digits and letters
input_string = '56aAww1984sktr235270aYmn145ss785fsq31D0'
 
# Extract numeric characters and alphabetic characters separately
digits_only = ''.join([char for char in input_string if char.isdigit()])
letters_only = ''.join([char for char in input_string if char.isalpha()])
 
# Display the separated substrings
print("Extracted digits substring:", digits_only)
print("Extracted letters substring:", letters_only)
 
# Convert even digits to their ASCII decimal values
even_digit_ascii_values = [ord(str(int(digit))) for digit in digits_only if int(digit) % 2 == 0]
print("ASCII values of even digits:", even_digit_ascii_values)
 
# Convert uppercase letters to their ASCII decimal values
uppercase_letter_ascii_values = [ord(letter) for letter in letters_only if letter.isupper()]
print("ASCII values of uppercase letters:", uppercase_letter_ascii_values)
 
