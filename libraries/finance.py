# Author: William Wadsworth
# Date: 6.16.2024
# 
# About:
#     This is the implementation file for the Python finance class

from typing import Tuple, Literal

class Finance:
    """A utility class for various financial calculations."""
    
    # pre-condition: monetary total parameter should be initialized to a non-zero and non-negative
    #                float 
    # post-condition: the minimum number of coins is returned in a tuple (quarters, dimes, nickels,
    #                 then pennies). If the total value parameter is less than or equal to 0, an 
    #                 error is output and a tuple consisting of four -1s is returned
    @staticmethod
    def findCoinTotal(total: float) -> Tuple[int,int,int,int]:
        """Returns the minimum number of coins given a monetary value."""
        
        # check amount, must be > 0
        if total <= 0:
            raise ValueError("invalid total, amount must be greater than 0")

        # convert dollars to cents to avoid floating-point issues
        total = round(total * 100)

        # how many QUARTERS in starting amount
        quarters = total // 25
        # calculate new total without number of quarters
        total %= 25

        # how many DIMES in updated amount
        dimes = total // 10
        total %= 10

        # how many NICKLES in updated amount
        nickels = total // 5
        #total %= 5

        # how many PENNIES in remaining amount
        pennies = total % 5

        return quarters, dimes, nickels, pennies
    
    # pre-condition: principal amount, interest rate, interest rate change, length of time, and
    #                deposit must all be initialized to positive non-zero floats
    # post-condition: the table is output detailing the total amount invested and the value of the
    #                 investment for each time step
    @staticmethod
    def genInvestmentTable(
        principal_amount: float,
        interest_rate: float,
        time: float,
        deposit: float,
        interest_rate_change: float,
        output: Literal["CONSOLE","FILE"]
    ) -> None:
        """Generates an investment table."""
        
        # ensure all numerical params are of valid types
        if not all(isinstance(param, (int, float)) for param in [principal_amount, interest_rate, interest_rate_change, time, deposit]):
            raise TypeError("All parameters must be numeric types (int or float)")

        # convert time (in years) to months
        time_months = int(time * 12)
        # loop control var
        t = 1
        # var to hold current investment value
        value_of_investment = 0.0
        # var to keep track of the month for changing interest
        count = 0

        # print table header
        table_title = f"{'Investment Table':^35}\n"
        table_header = f"{'Month':>8} | {'Total Invested ($)':>18} | {'Value of Investment ($)':>22}\n"
        table_separator = ('-' * 50) + '\n'
        
        investment_table: str = table_title + table_header + table_separator

        # table for changing interest
        while t <= time_months:
            # A = p + (p*r*t) + (t*d)
            value_of_investment = principal_amount + (principal_amount * interest_rate * t / 12) + (t * deposit)
            investment_table += f"{t:8} | {t * deposit:18.2f} | {value_of_investment:22.2f}\n"
            t += 1

            # for changing interest
            if interest_rate_change != 0:
                count += 1
                if count % 12 == 0:
                    interest_rate += interest_rate_change

        #print('-' * 50)
        end_msg = f"\nYour capital gain will be ${value_of_investment-principal_amount:.2f} in {time:.2f} years\n"
    
        if output == "CONSOLE":
            print(investment_table+end_msg)
        else:
            with open("investment_table",'w') as file:
                file.write(investment_table+end_msg)
    
    # pre-condition: each dollar denomination count must be initialized to a positive non-zero
    #                integer
    # post-condition: the monetary total is calculated and returned
    @staticmethod
    def moneyCalculator(
        count_1: int,
        count_5: int,
        count_10: int,
        count_20: int,
        count_50: int,
        count_100: int
    ) -> int:
        """Calculate how much money based on USD denominations (1,5,10,20,50,100)."""
        
        # ensure all parameters are integers
        if not all(isinstance(i, int) for i in [count_1, count_5, count_10, count_20, count_50, count_100]):
            raise TypeError("All denomination counts must be integers")

        return count_1 + (5*count_5) + (10*count_10) + (20*count_20) + (50*count_50) + (100*count_100)
    
