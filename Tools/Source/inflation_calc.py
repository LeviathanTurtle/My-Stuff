
# 
# 
# 

from os import getenv
from dotenv import load_dotenv
from requests import post

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
API_KEY = getenv("BLS_API_KEY")
if not API_KEY:
    raise ValueError("API key not found.")

# pre-condition: 
# post-condition: 
def get_cpi(year: int) -> float:
    """Fetch the CPI for a given year using the BLS API."""
    
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-Type": "application/json"}
    
    # CPI-U series ID for all urban consumers (not seasonally adjusted)
    series_id = "CUUR0000SA0"

    # Request payload with parameters
    payload = {
        "seriesid": [series_id],
        "startyear": str(year),
        "endyear": str(year),
        "registrationkey": API_KEY
    }

    response = post(url, json=payload, headers=headers)
    data = response.json()
    #print(f"Response: {response}")
    #print(f"data: {data}")
    
    # Check for errors
    if data['status'] != 'REQUEST_SUCCEEDED':
        raise ValueError("Failed to fetch CPI data.")
    
    # Extract CPI value
    try:
        cpi = float(data['Results']['series'][0]['data'][0]['value'])
    except (KeyError, IndexError, ValueError):
        raise ValueError("CPI data not available for the specified year.")
    
    return cpi

# pre-condition: 
# post-condition: 
def inflation_adjusted_amount(amount: float, from_year: int, to_year: int) -> float:
    """Adjusts an amount of money from one year to its equivalent value in another year, accounting
    for inflation based on CPI data from the BLS API."""
    
    cpi_from_year = get_cpi(from_year)
    cpi_to_year = get_cpi(to_year)
    
    # Adjust the amount for inflation
    return amount * (cpi_to_year/cpi_from_year)

def main():
    try:
        original_amount = 150000.0
        from_year = 1989
        to_year = 2024
        adjusted_amount = inflation_adjusted_amount(original_amount, from_year, to_year)
        print(f"${original_amount:,} in {from_year} is equivalent to ${adjusted_amount:,.2f} in {to_year}.")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()

