/*
 * plane.h
 *
 *
*/

#include <iostream>

using namespace std;

struct flight_distance {
    string start = "";
    string stop = "";
    int distance;
};
struct gate {
    char concourse;
    int gate_number;
};


class plane {
    private:
        // monetary
        int leasing_cost; // per month, setup in plane init

        // flying
        int max_fly_speed; // setup in plane init
        int current_fly_speed;

        // maintenance, fuel
        int hours_operated;
        int maintenance_cost; // setup in plane init
        int max_fuel_capacity; // setup in plane init

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

        // tail number 
        string tail_nunber; // setup in plane init

        // flight number (CA1234)
        enum flight_number;
        // needs get/set
    
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
        int GET_actual_flight_time();
        string GET_current_operation();
        gate GET_arrival_gate();
        gate GET_departure_gate();
        flight_distance GET_flight_path();
        string GET_tail_number();

        // set methods
        void SET_current_fly_speed(int);
        void SET_hours_operated(int);
        void SET_current_passengers(int);
        void SET_current_location(string);
        void SET_destination(string);
        void SET_departure_time(int);
        void SET_arrival_time(int);
        void SET_projected_flight_time(int);
        void SET_actual_flight_Time(int);
        void SET_current_operation(string);
        void SET_arrival_gate(gate);
        void SET_departure_gate(gate);
        void SET_flight_path(flight_distance);
        void SET_tail_number(string);

        // bool checks
        bool maintenance_check();
        bool is_valid_destination();

        // flight
        void takeoff();
        void landing();

        // info, debug
        void get_plane_info();
}