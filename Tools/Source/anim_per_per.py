# animals per person of a country's population

# 
# 
# 

from typing import Union
from requests import get, RequestException

def get_country_population(country_name: str) -> Union[None, int]:
    """Fetches the population of a specified country using the World Bank API."""
    
    # World Bank API endpoint for country population
    api_url = "http://api.worldbank.org/v2/country/{}/indicator/SP.POP.TOTL?format=json"

    # Attempt to fetch data using the World Bank API
    try:
        # Convert country name to ISO-3166 alpha-3 code using REST Countries API
        iso_code_url = f"https://restcountries.com/v3.1/name/{country_name}"
        response = get(iso_code_url)
        response.raise_for_status()
        country_data = response.json()
        country_code = country_data[0]['cca3']

        # Fetch population data
        response = get(api_url.format(country_code))
        response.raise_for_status()
        population_data = response.json()

        # Extract the most recent population data
        for record in population_data[1]:
            if record and record['value']:
                return int(record['value'])

        raise ValueError("Population data not found for the specified country.")

    except RequestException as e:
        print(f"Error fetching population data: {e}")
    except (IndexError, KeyError, ValueError) as e:
        print(f"Error processing data: {e}")

    return None

def main():
    country = input("Enter the name of the country: ").strip()
    try:
        animal_population = int(input("Enter the population of the animal: "))
    except ValueError:
        print("Invalid input. Animal population must be a number.")
        return

    # Fetch the population of the country
    population = get_country_population(country)

    if population:
        # Calculate animals per person
        if population == 0:
            print("The country's population is zero, cannot calculate.")
        else:
            # todo: find a better way to get animal_population
            animals_per_person = animal_population / population
            print(f"In {country}, there are approximately {animals_per_person:.2f} animals per person.")
    else:
        print("Unable to retrieve population data. Please try again.")

if __name__ == "__main__":
    main()

# note: https://systemanaturae.org/ ?