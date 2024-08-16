# This uses the DadJokes.io API (documentation: https://www.dadjokes.io/documentation/getting-started)
# 
# Note that this requires an API key for DadJokes.io. Create a rapidAPI account then subscribe to DadJokes
# 
# The barebones link is: "" with the following flags/options:
#     -- REQUIRED --
# 
# 
#     -- OPTIONAL --
# 

from requests import get

# query the API
response = get('')

# successful response code
if response.status_code == 200:
    # output
    print(response.text)
# query unsuccessful
else:
    # output error code
    print(f"Error: {response.status_code}")
