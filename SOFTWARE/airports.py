#from wheres_your_md import my_dick

#
# airport class
# 

class Airport:
    # airport object initialization
    def __init__(self):
        # name
        self.name = ""
        self.iata = ""
        # location
        self.city = ""
        self.state = ""
        self.latitude = 0
        self.longitude = 0
        # people
        self.metro_population = 0
        # money
        self.takeoff_fee = 0
        self.landing_fee = 0
        self.gas_price = 0
        # can it have flights to paris
        # default of false because only 1 (?) will be True
        self.paris_acceptable = False
        # how many gates does the airport have?
        self.num_gates = 0
    
    
    # description: determines if the airport can perform maintenance on
    #              aircraft (whether or not the airport is a hub)
    # pre-condition: Airport object must be initialized and IATA codes must be 
    #                not null/empty
    # post-condition: True/False is returned 
    def is_hub(self):
        # list of hub iata codes
        hub_iatas = {"ATL","DFW","DEN","ORD"}
        # if the iata code matches, return true, otherwise false
        return self.iata in hub_iatas
    
    
    # description: determines the number of gates at the airport (hubs have 11,
    #              others have one gate per million people in their metro 
    #              population, with a max of 5)
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
            #     the second
            # `//` -> floor divison
    
    
    # description: determines if the airport can fly to paris
    # pre-condition: Airport object must be initialized 
    # post-condition: If the aiport is Atlanta, its paris_acceptable bool is
    #                 updated, and True is returned. Otherwise, False is
    #                 returned 
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
        pass
    
    
    # description: function for "factory"
    # pre-condition:  
    # post-condition: 
    def factory(self):
        pass

