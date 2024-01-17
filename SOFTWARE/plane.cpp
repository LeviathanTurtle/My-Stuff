/*
 * plane.cpp
 *
 *
*/

#include "plane.h"
#include <iostream>

using namespace std;

// 15
void plane::init_Boeing_737_600()
{
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
    // typical cruise speed, 35k ft
    max_fly_speed = .82 // mach
    max_fuel_capacity = 5790 // us gal, 21918 L

    // typical 1-class config ?
    max_possible_passengers = 160;
    // typical 2-class config ?
    //max_possible_passengers = 141;
}



/*
class plane {
    private:
        // flying
        int max_fly_speed; // setup in plane init
        int current_fly_speed;

        // maintenance, fuel
        int hours_operated;
        int maintenance_cost; // setup in plane init
        int max_fuel_tank; // setup in plane init

        // passengers
        int max_possible_passengers; // setup in plane init
        int current_passengers;

        // location -> current/destination
        string current_location;
        string destination;
        int destination_distance;

        // arrival/departure
        int departure_time;
        int arrival_time;
        // CHANGE FROM INT TO SOMETHING ELSE

        // flight time
        int projected_flight_time;
        int actual_flight_time;

        // what the plane is currently doing
        string current_operation;

        // gates
        gate arrival;
        gate depature;

        // distance between locations
        flight_distance flight_path;
    
    public:
        // plane object initializations
        void init_Boeing_737_600();
        void init_Boeing_737_800();
        void init_Airbus_A200_100();
        void init_Airbus_A220_300();

        // get methods
        int GET_current_fly_speed();
        int GET_hours_operated();
        int GET_current_passengers();
        string GET_current_location();
        string GET_destination();
        int GET_departure_time();
        int GET_arrival_time();
        int GET_projected_flight_time();
        int GET_actual_flight_Time();
        string GET_current_operation();
        gate GET_arrival_gate();
        gate GET_departure_gate();
        flight_distance GET_flight_path();

        // set methods
        int SET_current_fly_speed();
        int SET_hours_operated();
        int SET_current_passengers();
        string SET_current_location();
        string SET_destination();
        int SET_departure_time();
        int SET_arrival_time();
        int SET_projected_flight_time();
        int SET_actual_flight_Time();
        string SET_current_operation();
        gate SET_arrival_gate();
        gate SET_departure_gate();
        flight_distance SET_flight_path();

        // bool checks
        bool maintenance_check();
        bool is_valid_destination();

        // flight
        void takeoff();
        void landing();
        void update_destination();

        // info, debug
        void get_plane_info();
}
*/