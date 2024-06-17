# Author: William Wadsworth
# Date: 6.16.2024
# 
# About:
#     This is the implementation file for the finance class

from typing import Tuple

class finance:
    """idk"""
    
    # function to return the minimum number of coins given a monetary value
    # pre-condition: monetary total parameter should be initialized to a non-zero and non-negative
    #                float 
    # 
    # post-condition: the minimum number of coins is returned in a tuple (quarters, dimes, nickels,
    #                 then pennies). If the total value parameter is less than or equal to 0, an 
    #                 error is output and a tuple consisting of four -1s is returned
    def findCoinTotal(total: float) -> Tuple[int,int,int,int]:
        # check amount, must be > 0
        if total <= 0:
            raise ValueError("invalid total, amount must be greater than 0")

        # convert dollars to cents to avoid floating-point issues
        total = int(total * 100 + 0.5)  # adding 0.5 to round correctly

        # QUARTERS
        # how many quarters in starting amount
        q = total // 25
        # calculate new total without number of quarters
        total -= q * 25

        # DIMES
        # how many dimes in updated amount
        d = total // 10
        total -= d * 10

        # NICKELS
        # how many nickels in updated amount
        n = total // 5
        total -= n * 5

        # PENNIES
        # how many pennies in remaining amount
        p = total

        return q, d, n, p
    
    # function to calculate how much money based on USD denominations (1,5,10,20,50,100)
    # pre-condition: each dollar denomination count must be initialized to a positive non-zero
    #                integer
    # 
    # post-condition: the monetary total is calculated and returned
    def moneyCalculator(count_1: int, count_5: int, count_10: int, count_20: int, count_50: int, count_100: int) -> int:
        # ensure all parameters are integers
        if not all(isinstance(i, int) for i in [count_1, count_5, count_10, count_20, count_50, count_100]):
            raise TypeError("All denomination counts must be integers")

        return count_1 + (5 * count_5) + (10 * count_10) + (20 * count_20) + (50 * count_50) + (100 * count_100)
    
    # function to generate an investment table
    # pre-condition: principal amount, interest rate, interest rate change, length of time, and
    #                deposit must all be initialized to positive non-zero floats
    # 
    # post-condition: the table is output detailing the total amount invested and the value of the
    #                 investment for each time step
    def genInvestmentTable(principal_amount: float, interest_rate: float, time: float, deposit: float, interest_rate_change: float = 0) -> None:
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
        print(f"{'Investment Table':^35}\n")
        print(f"{'Month':>8} | {'Total Invested ($)':>18} | {'Value of Investment ($)':>22}")
        print('-' * 50)

        # table for changing interest
        while t <= time_months:
            # A = p + (p*r*t) + (t*d)
            value_of_investment = principal_amount + (principal_amount * interest_rate * t / 12) + (t * deposit)
            print(f"{t:8} | {t * deposit:18.2f} | {value_of_investment:22.2f}")
            t += 1

            # for changing interest
            if interest_rate_change != 0:
                count += 1
                if count % 12 == 0:
                    interest_rate += interest_rate_change

        print('-' * 50)
        print(f"\nYour capital gain will be ${value_of_investment-principal_amount:.2f} in {time:.2f} years\n")

