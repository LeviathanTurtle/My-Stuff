# Shows a random dog photo from the Dog API (https://dog.ceo/dog-api/)

from requests import get
from PIL import Image
from io import BytesIO
from json import loads

def main():
    # query the API
    response = get('https://dog.ceo/api/breeds/image/random')

    # successful response code
    if response.status_code == 200:
        # Parse the JSON response
        data = loads(response.text)
        
        # Get the image URL from the JSON response
        image_url = data['message']
        
        # Send a GET request to the image URL
        image_response = get(image_url)
        
        # Check if the image request was successful
        if image_response.status_code == 200:
            # Open the image
            image = Image.open(BytesIO(image_response.content))
            
            # Show the image
            image.show()
        else:
            print(f"Failed to fetch the image. Status code: {image_response.status_code}")
    # query unsuccessful
    else:
        # output error code
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    main()