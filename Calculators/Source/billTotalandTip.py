# 
# Calculates the final bill for a restaurant
# WILLIAM WADSWORTH
# 

def billCalculator(
    food: float,
    tax: float,
    tip_percent: float = 0.20,
    num_members: int = 1,
    split: bool = False
) -> float:
    if split:
        pre_total = food+tax + ((food+tax) * tip_percent)
        return round(pre_total/num_members,2)
    else:
        pre_total = food+tax
        return round(pre_total+(pre_total*tip_percent),2)

def main():
    # get base food bill
    food_bill = float(input("Enter the price of the meal (no tax/tip): $"))
    while food_bill <= 0.0: # input validation
        food_bill = float(input("Please enter a valid total (e.g. 17.33): $"))
    
    # get amount of tax
    tax = float(input("Enter the taxed amount: $"))
    while tax < 0.0: # input validation
        tax = float(input("Please enter a valid tax (e.g. 1.3, 0): $"))
    
    # get how much the user wants to tip
    tip_percent = float(input("Enter the tip percentage in decimal form: "))
    while tip_percent < 0.0: # input validation
        tip_percent = float(input("Please enter a valid tip percentage (e.g. 0.23, 0): $"))
    
    # if the user is splitting the bill, this impacts the final calculation
    is_splitting: bool = True if input("Is the bill being split? [Y/n]: ") == 'Y' else False
    if is_splitting:
        num_members = int(input("How many people are splitting the bill? "))
        print(f"The bill total is ${billCalculator(food_bill,tax,tip_percent,num_members,split=True)} per {num_members} people")
    else:
        print(f"The bill total is ${billCalculator(food_bill,tax,tip_percent)}")

if __name__ == "__main__":
    main()