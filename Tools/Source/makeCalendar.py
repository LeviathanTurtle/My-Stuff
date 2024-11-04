# generates a calendar based on an input year and month

from calendar import month

year = int(input("Enter year: "))
month = int(input("Enter month: "))

cal = month(year,month)

print(cal)