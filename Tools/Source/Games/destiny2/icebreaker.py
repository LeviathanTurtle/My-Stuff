
# 
# William Wadsworth
# 
# for the icebreaker catalyst puzzle
# 

from itertools import permutations

def first_puzzle(x: int):
    results = []
    
    # generate all combinations of 4 numbers between 0 and 9
    for combination in permutations(range(10), 4): # ensure no repeats
        if sum(combination) == x:
            # add sorted tuple for uniqueness
            results.append(tuple(sorted(combination)))

    # remove duplicates by converting to a set, then back to a list
    results = list(set(results))
    return results

def second_puzzle(x: int):
    results = []
    
    # generate all combinations of 4 numbers between 0 and 9
    for combination in permutations(range(10), 4): #  ensure no repeats
        product = combination[0] * combination[1] * combination[2] * combination[3]
        if product == x:
            # add sorted tuple for uniqueness
            results.append(tuple(sorted(combination)))

    # remove duplicates by converting to a set, then back to a list
    results = list(set(results))
    return results

def third_puzzle(x: int):
    results = []
    
    # generate all possible sums of four numbers (0-9) using the addition function
    all_sums = {}
    # minimum sum for 4 numbers is 0+1+2+3 = 6, max is 9+8+7+6 = 30
    for y in range(6, 30):
        combinations = first_puzzle(y)
        if combinations:
            all_sums[y] = combinations

    # iterate over possible values of y and z, where y * z = x
    for y in all_sums:
        z = x // y
        if x % y != 0: # ensure exact division
            continue
        if z in all_sums: # check if z is a valid sum
            for combo_y in all_sums[y]:
                for combo_z in all_sums[z]:
                    results.append((combo_y, combo_z))

    return results

def main():
    number_1 = int(input("First puzzle number(s): "))
    print(first_puzzle(number_1))

    number_2 = int(input("Second puzzle number(s): "))
    print(second_puzzle(number_2))
    
    number_3 = int(input("Third puzzle number(s): "))
    results = third_puzzle(number_3)
    for y, z in results:
        print(f"y: {y} (sum = {sum(y)}), z: {z} (sum = {sum(z)})")

if __name__ == "__main__":
    main()