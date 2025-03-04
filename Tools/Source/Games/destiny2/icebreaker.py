# 
# for the icebreaker catalyst puzzle
# 

from itertools import permutations, combinations

def first_puzzle(x: int):
    #if x < 0 or x > 36:  # Impossible sum range for 4 digits 0-9
    #    return []

    results = []
    # Generate all combinations of 4 numbers between 0 and 9
    for combination in permutations(range(10), 4):  # Permutations ensure no repeats
        if sum(combination) == x:
            results.append(tuple(sorted(combination)))  # Add sorted tuple for uniqueness

    # Remove duplicates by converting to a set, then back to a list
    results = list(set(results))
    return results

def second_puzzle(x: int):
    #if x < 0:  # Products can't be negative
    #    return []

    results = []
    # Generate all combinations of 4 numbers between 0 and 9
    for combination in permutations(range(10), 4):  # Permutations ensure no repeats
        product = combination[0] * combination[1] * combination[2] * combination[3]
        if product == x:
            results.append(tuple(sorted(combination)))  # Add sorted tuple for uniqueness

    # Remove duplicates by converting to a set, then back to a list
    results = list(set(results))
    return results

def third_puzzle(x: int):
    #if x < 0:  # Products can't be negative
    #    return []

    results = []
    
    #print("test1")
    # Generate all possible sums of four numbers (0-9) using the addition function
    all_sums = {}
    for y in range(6, 30):  # Minimum sum for 4 numbers is 0+1+2+3 = 6, max is 9+8+7+6 = 30
        #print("test2")
        combinations = first_puzzle(y)
        if combinations:
            all_sums[y] = combinations

    # Iterate over possible values of y and z where y * z = x
    for y in all_sums:
        #print("test3")
        z = x // y
        if x % y != 0:  # Ensure exact division
            continue
        if z in all_sums:  # Check if z is a valid sum
            #print("test4")
            for combo_y in all_sums[y]:
                #print("test5")
                for combo_z in all_sums[z]:
                    #print("test6")
                    results.append((combo_y, combo_z))

    return results

def main():
    number_1 = int(input("First puzzle number: "))
    print(first_puzzle(number_1))

    number_2 = int(input("Second puzzle number: "))
    print(second_puzzle(number_2))
    
    number_3 = int(input("Third puzzle number: "))
    results = third_puzzle(number_3)
    for y, z in results:
        print(f"y: {y} (sum = {sum(y)}), z: {z} (sum = {sum(z)})")

if __name__ == "__main__":
    main()