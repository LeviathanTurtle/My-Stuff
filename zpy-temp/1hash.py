# from: GeeksForGeeks: https://www.geeksforgeeks.org/python-hash-method/


# initializing objects
int_val = 4
str_val = 'MyAss'
flt_val = 24.56
 
# Printing the hash values.
# Notice Integer value doesn't change
# You'll have answer later in article.
print("The integer hash value is : " + str(hash(int_val)))
print("The string hash value is : " + str(hash(str_val)))
print("The float hash value is : " + str(hash(flt_val)))




# initializing objects
# tuple are immutable
tuple_val = (1, 2, 3, 4, 5)
 
# list are mutable
list_val = [1, 2, 3, 4, 5]
 
# Printing the hash values.
# Notice exception when trying
# to convert mutable object
print("The tuple hash value is : " + str(hash(tuple_val)))
print("The list hash value is : " + str(hash(list_val)))




# hash() for immutable tuple object
var = ('G','E','E','K')
print(hash(var))




l = [1, 2, 3, 4]
print(hash(l))




class Emp:
    def __init__(self, emp_name, id):
        self.emp_name = emp_name
        self.id = id
 
    def __eq__(self, other):
        # Equality Comparison between two objects
        return self.emp_name == other.emp_name and self.id == other.id
 
    def __hash__(self):
        # hash(custom_object)
        return hash((self.emp_name, self.id))
 
emp = Emp('Ragav', 12)
print("The hash is: %d" % hash(emp))
 
# We'll check if two objects with the same
# attribute values have the same hash
emp_copy = Emp('Ragav', 12)
print("The hash is: %d" % hash(emp_copy))
