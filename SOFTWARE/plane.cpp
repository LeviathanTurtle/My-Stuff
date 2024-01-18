/*
 * plane.cpp
 *
 *
*/

#include "plane.h"
#include <iostream>
#include <sys/time.h>

using namespace std;

// ----------------------------------------------------------------------------
// INITS

// 15
void plane::init_Boeing_737_600()
{
    // leasing cost (per month, usd)
    leasing_cost = 245000;

    // set maintenance cost
    
    // typical cruise speed, 35k ft
    max_fly_speed = 0.785 // mach
    max_fuel_capacity = 6875 // us gal, 26020 L

    // typical 1-class config
    max_possible_passengers = 132;
    // typical 2-class config
    //max_possible_passengers = 110;
}
// 15
void plane::init_Boeing_737_800()
{
    // leasing cost (per month, usd)
    leasing_cost = 270000;

    // set maintenance cost

    // typical cruise speed, 35k ft
    max_fly_speed = 0.785 // mach
    max_fuel_capacity = 6875 // us gal, 26020 L

    // typical 1-class config
    max_possible_passengers = 189;
    // typical 2-class config
    //max_possible_passengers = 162;
}
// 12
void plane::init_Airbus_A200_100()
{
    // leasing cost (per month, usd)
    leasing_cost = 192000;

    // set maintenance cost

    // typical cruise speed, 35k ft
    max_fly_speed = 0.785 // mach
    max_fuel_capacity = 5790 // us gal, 21918 L

    // typical 1-class config ?
    max_possible_passengers = 135;
    // typical 2-class config ?
    //max_possible_passengers = 116;
}
// 13
void plane::init_Airbus_A220_300()
{
    // leasing cost (per month, usd)
    leasing_cost = 228000;
    
    // set maintenance cost

    // typical cruise speed, 35k ft
    max_fly_speed = .82 // mach
    max_fuel_capacity = 5790 // us gal, 21918 L

    // typical 1-class config ?
    max_possible_passengers = 160;
    // typical 2-class config ?
    //max_possible_passengers = 141;
}

// ----------------------------------------------------------------------------
// GET

int plane::GET_current_fly_speed()
{
    return current_fly_speed;
}

int plane::GET_hours_operated()
{
    return hours_operated;
}

int plane::GET_current_passengers()
{
    return current_passengers;
}

string plane::GET_current_location()
{
    return current_location;
}

string plane::GET_destination()
{
    return destination;
}

int plane::GET_departure_time()
{
    return departure_time;
}

int plane::GET_arrival_time()
{
    return arrival_time;
}

int plane::GET_projected_flight_time()
{
    return projected_flight_time;
}

int plane::GET_actual_flight_time()
{
    return projected_flight_time;
}

string plane::GET_current_operation()
{
    return current_operation;
}

gate plane::GET_arrival_gate()
{
    return arrival;
}

gate plane::GET_departure_gate()
{
    return departure;
}

flight_distance plane::GET_flight_path()
{
    return flight_distance;
}

string plane::GET_tail_number()
{
    return tail_nunber;
}

// ----------------------------------------------------------------------------
// SET

void plane::SET_current_fly_speed(int current_fly_speed_in)
{
    current_fly_speed = current_fly_speed_in;
}

void plane::SET_hours_operated(int hours_operated_in)
{
    hours_operated = hours_operated_in;
}

void plane::SET_current_passengers(int current_passengers_in)
{
    current_passengers = current_passengers_in;
}

void plane::SET_current_location(string current_location_in)
{
    current_location = current_location_in;
}

void plane::SET_destination(string destination_in)
{
    destination = destination_in;
}

void plane::SET_departure_time(int departure_time_in)
{
    departure_time = departure_time_in;
}

void plane::SET_arrival_time(int arrival_time_in)
{
    arrival_time = arrival_time_in;
}

void plane::SET_projected_flight_time(int projected_flight_time_in)
{
    projected_flight_time = projected_flight_time_in;
}

void plane::SET_actual_flight_Time(int actual_flight_time_in)
{
    actual_flight_time = actual_flight_time_in;
}

void plane::SET_current_operation(string current_operation_in)
{
    current_operation = current_operation_in;
}

void plane::SET_arrival_gate(gate arrival_in)
{
    arrival = arrival_in;
}

void plane::SET_departure_gate(gate departure_in)
{
    departure = departure_in;
}

void plane::SET_flight_path(flight_distance flight_path_in)
{
    flight_path = flight_path_in;
}

void plane::SET_tail_number(string tail_nunber_in)
{
    tail_nunber = tail_nunber_in;
}

// ----------------------------------------------------------------------------
// BOOLS

bool plane::maintenance_check()
{
    if(GET_hours_operated() >= 200)
        return true;
    else
        return false;
}

bool plane::is_valid_destination()
{
    if(/*distance between 2 destinations is > 150 mi*/)
        return true;
    else
        return false;
}

// ----------------------------------------------------------------------------
// FLIGHT

// time checks should be sprinkled in, i.e. departTime-55
void plane::takeoff()
{
    // ------------------------------------------------------------------------
    // BOARDING

    SET_current_operation("Boarding");

    SET_departure_gate(/*gate E11*/);
    SET_destination("California");
    //SET_flight_nunber("CA1234");
    SET_departure_time(/*11:00*/);
    SET_projected_flight_time(/*flight eta*/);

    SET_current_passengers(/*constrained rng*/);

    SET_current_operation("Taking off");
    // gettimeofday stuff ...
    
    SET_current_operation("In flight");
}

void plane::landing()
{
    SET_current_operation("Arriving at gate");
    // gettimeofday stuff...

    // add last flight's hours to total flight hours
    SET_hours_operated(GET_hours_operated()+GET_actual_flight_time());

    // ------------------------------------------------------------------------
    // UNLOADING

    SET_current_operation("Dismbarking");
    // time check, record time
    
    SET_current_location(/*current location*/);

    // ------------------------------------------------------------------------
    // MAINTENANCE

    SET_current_operation("Performing maintenance");

    // check flight hours for maintenance, default false
    // set destination based on result
    bool needs_maintenance = false;
    if(maintenance_check()) {
        needs_maintenance = true;
        SET_destination(/*rng from array of hubs*/);
    }
    else
        SET_destination(/*rng from array of destinations*/);
}

// ----------------------------------------------------------------------------
// DEBUG, INFO

void plane::get_plane_info()
{
    
}