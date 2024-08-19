# This uses the Random User Generator API (documentation: https://randomuser.me/documentation)
# 
# The base link is: https://randomuser.me/api/ with the following options:
#     -- OPTIONAL --
# nat      : specify a nationality
# inc      : only include certain data
# exc      : exclude certain data
# page     : request multiple pages
# results  : number of users to generate
# gender   : specify user gender (male, female)
# seed     : a way to get the same set of random users
# format   : specify response format
# lego     : returns a lego avatar

from requests import get


def main():
    # query the API
    response = get('https://randomuser.me/api/')

    # successful response code
    if response.status_code == 200:
        # store response in var
        data = response.json()
        
        # store user info in corresponding vars
        user = data['results'][0]
        name = user['name']
        location = user['location']
        email = user['email']
        login = user['login']
        dob = user['dob']
        registered = user['registered']
        phone = user['phone']
        cell = user['cell']
        id_info = user['id']
        #picture = user['picture']
        nat = user['nat']
        
        print("Name:", name['title'], name['first'], name['last'])
        print("Location:", location['street']['number'], location['street']['name'], location['city'], location['state'], location['country'], location['postcode'])
        print("Email:", email)
        print("Username:", login['username'])
        print("Password:", login['password'])
        print("Date of Birth:", dob['date'], "Age:", dob['age'])
        print("Registered:", registered['date'])
        print("Phone:", phone)
        print("Cell:", cell)
        print("ID:", id_info['name'], id_info['value'])
        #print("Picture:", picture['large'])
        print("Nationality:", nat)
    # query unsuccessful
    else:
        # output error code
        print(f"Error: {response.status_code}\nFailed to retrieve user data")


if __name__ == "__main__":
    main()