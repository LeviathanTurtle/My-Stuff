# This uses the JokeAPI (documentation: https://sv443.net/jokeapi/v2/)
# 
# The barebones link is: "https://v2.jokeapi.dev/joke/" with the following flags/options:
#     -- REQUIRED --
# category - Any, Programming, Miscellaneous, Dark, Pun, Spooky, Christmas
# lang     - select language
# format   - response format
# type     - single or twopart
# 
#     -- OPTIONAL --
# blacklistFlags - topics to exclude
# contains       - get a joke that contains the search string
# idRange        - get jokes between a range of IDs
# amount         - amount of jokes to generate
# 
# Note that if you want to include more than one option, separate using a comma.

from requests import get


def main():
    # ask the user for the amount of jokes they'd like to generate
    num_jokes = input("How many jokes would you like to generate: ")

    # query the API
    response = get(f'https://v2.jokeapi.dev/joke/Any?format=txt&amount={num_jokes}')

    # successful response code
    if response.status_code == 200:
        # output
        print(response.text)
    # query unsuccessful
    else:
        # output error code
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    main()