# https://zenquotes.io/
# 
# This uses the ZenQuotes.io API (documentation: https://docs.zenquotes.io/zenquotes-documentation)
# 
# The link template is: https://zenquotes.io/api/[mode]/[key]?option1=value&option2=value
# where
#     -- REQUIRED --
# https://zenquotes.io/api =  ZenQuotes API URL
# [mode]                   =  Retrieval type [quotes, today, author, random]
# 
#     -- OPTIONAL --
# [key]                    =  API key for use with premium subscriptions, be sure to obfuscate or
#                             hide this in your source code to prevent hijacking. Optional
# [options]                =  Additional options. Optional
# 
# The response will be formatted as a JSON array:
# q: quote text
# a: author name
# i: author image (key required)
# h: pre-formatted HTML quote

from requests import get


def main():
    # query the API
    response = get('https://zenquotes.io/api/random')

    # successful response code
    if response.status_code == 200:
        # store response in var
        json_response = response.json()
        
        # check that the response is a list and contains at least 1 element
        if isinstance(json_response, list) and len(json_response) > 0:
            # take out the first element
            quote_data = json_response[0]
            # extract the quote, with a default message if not found
            quote = quote_data.get('q', 'No quote found')
            # extract the quote author, with a default if not found
            author = quote_data.get('a', 'Unknown author')
            # output result
            print(f'"{quote}" - {author}')
        else:
            # in case no quote was found
            print("No quote found in the response.")
    # query unsuccessful
    else:
        # output error code
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    main()