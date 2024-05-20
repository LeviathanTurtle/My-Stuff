from rembg import remove
from PIL import Image

input = Image.open('raccoon.jpg')
output = remove(input)
output.save('raccoon-out.png')