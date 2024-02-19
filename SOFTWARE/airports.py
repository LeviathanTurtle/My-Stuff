# Team: Foobar
# Teammates: Anthony Cox, Corey Lawrence, Dylan Hudson, Parker Blue, Will Wadsworth, Zach Christopher
# Authors: Anthony Cox, Will Wadsworth
# Date: 2/16/2024
#
# Description:
#   This module defines and implements the model class `Airport`

#from wheres_your_md import my_dick
import queue


class Airport:
    # airport object initialization
    def __init__(
        self, name: str, iata: str, city: str, state: str, latitude: int, longitude: int, 
        metro_population: int, takeoff_fee: int, landing_fee: int, gas_price: int,
        paris_acceptable: bool, num_gates: int
        ):
        # name
        self.name = name
        self.iata = iata
        # location
        self.city = city
        self.state = state
        self.latitude = latitude
        self.longitude = longitude
        # people
        self.metro_population = metro_population
        # money
        self.takeoff_fee = takeoff_fee
        self.landing_fee = landing_fee
        self.gas_price = gas_price
        # can it have flights to paris
        # default of false because only 1 (?) will be True
        self.paris_acceptable = paris_acceptable
        # how many gates does the airport have?
        self.num_gates = num_gates
    
    
    # description: determines if the airport can perform maintenance on aircraft (whether or not
    #              the airport is a hub)
    # pre-condition: Airport object must be initialized and IATA codes must be  not null/empty
    # post-condition: True/False is returned 
    def is_hub(self):
        # list of hub iata codes
        hub_iatas = {"ATL","DFW","DEN","ORD"}
        # if the iata code matches, return true, otherwise false
        return self.iata in hub_iatas
    
    
    # description: determines the number of gates at the airport (hubs have 11, others have one
    #              gate per million people in their metro population, with a max of 5)
    # pre-condition: Airport object must be initialized 
    # post-condition: The airport's number of gates is updated 
    def determine_gates(self):
        if self.is_hub():
            # airport is a hub -- 11 gates
            self.num_gates = 11
        else:
            # assign a gate per 1million people in the metro pop.
            self.num_gates = min(self.metro_population // 1000000, 5)
            # min() will take the first parameter as long as it does not exceed
            #     the second. If the first is higher, it takes the second parameter
            # `//` -> floor divison
    
    
    # description: determines if the airport can fly to paris
    # pre-condition: Airport object must be initialized 
    # post-condition: If the aiport is Atlanta, its paris_acceptable bool is updated, and True is
    #                 returned. Otherwise, False is returned 
    def is_paris(self):
        if self.iata == "ATL":
            self.paris_acceptable = True
            return True
        else:
            return False
    

    # description: function for tarmac queue
    # pre-condition:  
    # post-condition: 
    def tarmac_queue(self):
        # create queue object
        tarmac = queue.Queue()
        
        # plane lands -> check for gate
        # if gate:
        #     goto gate
        # else:
        #     add to queue <- this here
    
    
    # description: function for "factory"
    # pre-condition:  
    # post-condition: 
    def factory(self):
        pass
    # idk what this is I forgot

