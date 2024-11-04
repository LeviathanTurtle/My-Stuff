# This generates an insult from the Evil Insult API (https://evilinsult.com/)
# 
# The base link is: https://opentdb.com/api.php
# where
#     -- OPTIONAL --
# lang  : language to use (default english)
# type  : response format (text, XML, JSON, or default plain text)

from requests import get

def main():
    # query the API
    response = get('https://evilinsult.com/generate_insult.php')

    # successful response code
    if response.status_code == 200:
        print(response.text)
    # query unsuccessful
    else:
        # output error code
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    main()