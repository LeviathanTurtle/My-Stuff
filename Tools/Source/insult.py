# This uses the Evil Insult API from https://evilinsult.com/
# 
# The base link is: https://opentdb.com/api.php
# where
#     -- REQUIRED --
# 
# 
#     -- OPTIONAL --
# lang  : language to use (default english)
# type  : response format (text, XML, JSON, or default plain text)

from requests import get

# query the API
response = get('https://evilinsult.com/generate_insult.php')

# successful response code
if response.status_code == 200:
    print(response.text)
# query unsuccessful
else:
    # output error code
    print(f"Error: {response.status_code}")
