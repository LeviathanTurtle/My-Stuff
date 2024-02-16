from enum import Enum

# 
# plane.py -- class
# 

# PLANE OBJECT CLASS
class Plane:
    # plane object initialization
    def __init__(self):
        # flying
        self.current_fly_speed = 0
        self.hours_operated = 0
        # passengers
        self.current_passengers = 0
        # location/destination
        self.current_location = ""
        self.destination = ""
        self.destination_distance = 0
        self.cardinal_direction = ""
        self.cardinal_degree = 0
        # time
        self.expec_depart_time = 0
        self.expec_arrival_time = 0
        self.actual_depart_time = 0
        self.actual_arrival_time = 0
        self.current_flight_time = 0
        # what is the plane doing
        self.current_operation = ""
        # gates
        self.arrival_gate = 0
        self.depart_gate = 0
        # direction
        self.flight_path = 0
        # plane info
        self.flight_num = Enum # this is enum
        
        self.in_flight = False
    
    # 15
    def __init_Boeing_737_600__(self):
        # constants, dependent on plane
        self.leasing_cost = 245000
        self.max_fly_speed = .785 # mach
        #self.maintenance_cost = 
        self.max_fuel_capacity = 6875 # us gal, 26020 L
        self.max_possible_passengers = 132
        self.tail_number = Enum # this is enum

    # 15
    def __init_Boeing_737_800__(self):
        # constants, dependent on plane
        self.leasing_cost = 270000
        self.max_fly_speed = .785 # mach
        #self.maintenance_cost = 
        self.max_fuel_capacity = 6875 # us gal, 26020 L
        self.max_possible_passengers = 189
        self.tail_number = Enum # this is enum

    # 12
    def __init_Airbus_A200_100__(self):
        # constants, dependent on plane
        self.leasing_cost = 192000
        self.max_fly_speed = .785 # mach
        #self.maintenance_cost = 
        self.max_fuel_capacity = 5790 # us gal, 26020 L
        self.max_possible_passengers = 135
        self.tail_number = Enum # this is enum
    
    # 13 
    def __init_Airbus_A220_300__(self):
        # constants, dependent on plane
        self.leasing_cost = 228000
        self.max_fly_speed = .82 # mach
        #self.maintenance_cost = 
        self.max_fuel_capacity = 5790 # us gal, 26020 L
        self.max_possible_passengers = 160
        self.tail_number = Enum # this is enum

# HINRANCES FUNCTION
def hindrance(day, plane):
    rng = None # create this at start
    
    if day == 3:
        for i in range(int(totalFlights*.25)):
            # bad weather, ground delay, extended flight time (rng between 1 min and flightTime)
            pass
    
    elif day == 5:
        if plane.cardinal_degree > 40 and plane.cardinal_direction == 'N':
            for i in range(int(totalFlights*.2)):
                # ground delay (ice) (rng between 10-45 min)
                pass
    
    elif day == 7:
        if plane.flight_path == 'E':
            plane.current_flight_time += plane.current_flight_time*.12
        elif plane.flight_path == 'W':
            plane.current_flight_time -= plane.current_flight_time*.12
    
    elif day == 9:
        for i in range(int(totalFlights*.05)):
            # gate delay (rng between 5-90 min)
            pass
    
    elif day == 11:
        # one (1) plane in major hub is towed away for maintenance for the day
        pass
    
    elif day == 14:
        if plane.cardinal_degree > 103 and plane.cardinal_direction == 'W':
            for i in range(int(totalFlights*.08)):
                # flight is cancelled, find a new flight
                pass

# FUNCTION TO GET DESTINATION
def get_destination(plane):
    pass

#    else:
#        catch error -- redundancy?

# OUTPUT FUNCTIONS
# this function is output information about the plane
def output_plane_info(plane):
    print("\n\nOutputing plane info...\n")
    print("Hours operated: ",plane.hours_operated)
    print("Flight number: ",plane.flight_num)
    
    if(plane.in_flight == True):
        print("\nFlight Information:\n")
        print("Current speed: ",plane.current_fly_speed)
        print("Passenger count: ",plane.current_passengers)
        print("Direction: ",plane.cardinal_direction)    
        print("Degree: ",plane.cardinal_degree)
        print("Current flight time: ",plane.current_flight_time)
    print("...done.\n")

def output_location_info(plane):
    print("\n\nOutputing locational info...\n")
    print("Current location: ",plane.current_location)
    print("Current operation: ",plane.current_operation)
    print("...done.\n")

# this function is output information for the flight
def output_flight_info(plane):
    print("\n\nOutputing flight info...\n")
    print("Destination: ",plane.destination)
    print("Destination distance: ",plane.destination_distance)
    print("Departure Gate: ",plane.depart_gate)
    print("Departure Time* (E): ",plane.expec_depart_time)
    print("Arrival Time* (E): ",plane.expec_arrival_time)
    print("Arrival Gate: ",plane.arrival_gate)
    print("Flight Path: ",plane.flight_path)
    print("...done.\n")

# this function is output that flight times were successfully updated
def update_flight_info(plane):
    print("\n\nUpdating flight info...\n")
    print("Actual depart time: ",plane.actual_depart_time)
    print("Actual arrival time: ",plane.actual_arrival_time)
    print("...done.\n")

# POPULATE PLANE OBJECTS
# THIS IS ONLY FOR TESTING

def populate_plane(plane):
    pass




# GLOBAL VARS
totalFlights = 0


# MAIN

# create plane objects
boeing737_600 = Plane()
boeing737_600.__init_Boeing_737_600__()
#boeing737_800 = Plane()
#boeing737_800.__init_Boeing_737_800__()
#airbusA200_100 = Plane()
#airbusA200_100.__init_Airbus_A200_100__()
#airbusA220_300 = Plane()
#airbusA220_300.__init_Airbus_A220_300__()


# repeat for each day over the 2 week period
# AT THE MOMENT, THIS ASSUMES ONE (1) FLIGHT PER DAY
for day in range(1,14):
    # where is the plane now and what is it doing?
    output_location_info(boeing737_600)
    
    # check hindrances for the day
    hindrance(day, boeing737_600)
    
    # prepare for next flight:
    # - log next flight info
    # - prepare for passengers
    # find next flight (destination)
    get_destination(boeing737_600)
    boeing737_600.current_operation = "Boarding"
    # set the departue gate to the arrival gate, because that has not been 
    # updated yet (arrival gate should be current gate)
    boeing737_600.depart_gate = boeing737_600.arrival_gate
    
    # take flight:
    boeing737_600.current_operation = "In Flight"
    # - log info
    
    
    # landing:
    # - log info
    boeing737_600.current_operation = "Landing"
    
    # go to gate
    # - log/update info
    boeing737_600.current_operation = "Arriving at destination"
    # ... time to gate ...
    boeing737_600.current_operation = "Disembarking"
    boeing737_600.current_operation = "Cleaning/Preparing for next flight"
    
    