# HOW MANY COINS ARE IN A DEPOSIT OF MONEY
# William Wadsworth
# Created: 
# Doctored: 10.25.2023
# 
# Python-ized: 4.23.2024
# Updated 8.18.2024: PEP 8 Compliance
# 
# [DESCRIPTION]:
# This program prompts the user to input a price, and the program will calculate and output the
# minimum amount of coins for each type (quarter, dime, nickel, penny) required to meet the price.
# 
# Note: does not always work, 21.31 does not include last penny. This will be fixed in the future.
# 
# [USAGE]:
# To run: python3 coinTotal.py
# 
# [EXIT/TERMINATING CODES]:
# 0 - program successfully completed a full execution
# 
# 1 - invalid amount


from sys import stderr, exit


def main():
    # --- GET TOTAL -----------------------------
    # prompt for starting amount, store input in variable
    total = float(input("How much money do you have: $"))
    # input validation, must be > 0
    if total <= 0:
        stderr.write("error: amount must be greater than 0.\n")
        exit(1)

    # --- QUARTERS ------------------------------
    # how many quarters in starting amount
    q = int(total // 0.25)
    # calculate new total without number of quarters
    total -= q*0.25
    # this process is repeated for each coin

    # --- DIMES ---------------------------------
    # how many dimes in updated amount
    d = int(total // 0.10)
    total -= d*0.10

    # --- NICKELS -------------------------------
    # how many nickels in updated amount
    n = int(total // 0.05)
    total -= n*0.05

    # --- PENNIES -------------------------------
    # how many pennies in remaining amount
    p = int(total // 0.01)
        
    # --- OUTPUT --------------------------------
    print(f"""\nYou can have as low as: {q+d+n+p} coins\n
          {'# of quarters: ':>3} {q}\n
          {'# of dimes: ':>3} {d}\n
          {'# of nickels: ':>3} {n}\n
          {'# of pennies: ':>3} {p}""")


if __name__ == "__main__":
    main()