###Chapter 1
import time
from PIL import Image

# Step 1: Generate a dynamic number based on the current time
current_time = int(time.time())
generated_number = (current_time % 100) + 50

# Adjust the number if it's even by adding 10
if generated_number % 2 == 0:
    generated_number += 10

# Display the final generated number
print("The dynamic number generated is:", generated_number)

# Step 2: Open the image file
img = Image.open('chapter1.jpg')
pixels = img.load()

# Step 3: Adjust the color values of all pixels using the generated number
width, height = img.size
total_red_value = 0

# Iterate through each pixel and adjust the red, green, and blue values
for x in range(width):
    for y in range(height):
        r, g, b = pixels[x, y]
        new_r = min(255, r + generated_number)  # Ensure values don't exceed 255
        new_g = min(255, g + generated_number)
        new_b = min(255, b + generated_number)
        pixels[x, y] = (new_r, new_g, new_b)

        # Accumulate the red pixel values to get the total sum
        total_red_value += new_r

# Step 4: Save the modified image to a new file
img.save('chapter1out.png')

# Step 5: Print the total of all red pixel values
print("Total of all red pixel values after modification:", total_red_value)
