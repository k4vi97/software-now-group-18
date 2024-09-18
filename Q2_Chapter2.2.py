### Chapter 2.2
 
# Function to decrypt a Caesar cipher text using a specified shift value
def decrypt_caesar_cipher(ciphertext, shift_value):
    decrypted_text = [] # List to store the decrypted characters
    for character in ciphertext:
        if character.isalpha():
            shifted_value = ord(character) - shift_value
            # Adjust for wraparound if needed
            if character.isupper():
                if shifted_value < ord('A'):
                    shifted_value += 26
            elif character.islower():
                if shifted_value < ord('a'):
                    shifted_value += 26
            # Append the decrypted character
            decrypted_text.append(chr(shifted_value))
        else:
            # Non-alphabetic characters remain unchanged
            decrypted_text.append(character)
    return ''.join(decrypted_text) # Join list into a single string and return
 
# Example cryptogram to be decrypted
ciphered_message = 'VZ FRYSVFU VZCNGVRAG NAQ N YVGGYR VAFRPHER V ZNXR ZVFGNXRF V NZ BHG BS PBAGEBY NAQNG GVZRF UNEQ GB UNAQYR OHG VS LBH PNAG UNAQYR ZR NG ZL JBEFG GURA LBH FHER NF URYYQBAG QRFREIR ZR NG ZL ORFG ZNEVYLA ZBAEBR'
 
# The shift key to decrypt the message (this value should be determined based on the problem or analysis)
shift_key_value = 13
 
# Decrypt the cryptogram using the Caesar cipher decryption function
decrypted_quote = decrypt_caesar_cipher(ciphered_message, shift_key_value)
print("Decrypted quote:", decrypted_quote)
 