#include <iostream>
using namespace std;

struct planeObj {
    string departDestination;
    string arriveDestination;
    int numPassengers;
    int hoursOperated;
    double percentOnTimeArrival;
    double percentOnTimeDeparture;
    // keys
    int tailNum;
    int flightNum;
    // location
    struct location {
        int degree;
        char direction;
    };
    char flightPath;
};


// 10
string hubs[] = {"Atlanta","Dallas","Denver","Chicago","Los Angeles","New York City","Las Vegas","Orlando","Miami","Charlotte"};


int main()
{
    planeObj plane;

    outputPlaneInfo(plane);

    for(int day=1; day<=10; day++)
        hindrances(day, plane);

    outputPlaneInfo(plane);

    return 0;
}

void hindrances(const int& day, planeObj& plane)
{
    int rng;

    switch(day) {
        case(3):
            for(int i=0; i<totalFlights*.25; i++)
                // bad weather, ground delay, extended flight time (rng between 1 min and flightTime)

        case(5):
            if(plane.location.degree > 40 && plane.location.direction == 'N')
                for(int i=0; i<totalFlights*.2; i++)
                    // ground delay (ice) (rng between 10-45 min)
        
        case(7):
            if(plane.flightPath == 'E')
                plane.flightTime += flightTime*.12;
            if(plane.flightPath == 'W')
                plane.flightTime -= flightTime*.12;
        
        case(9):
            for(int i=0; i<totalFlights*.05; i++)
                // gate delay (rng between 5-90 min)
        
        case(11):
            // one (1) plane in major hub is towed away for maintenance for the day

        case(13):
            if(plane.location.degree > 103 && plane.location.direction == 'W')
                for(int i=0; i<totalFlights*.08; i++)
                    // flight is cancelled, find a new flight
    }
}

void outputPlaneInfo(const planeObj& plane)
{
    cout << "tail number: " << plane.tailNum << endl;
    cout << "flight number (of the day): " << plane.flightNum << endl;
    cout << "departing from: " << plane.departDestination << endl;
    cout << "arriving at: " << plane.arriveDestination << endl;
    cout << "number of passengers on this flight: " << plane.numPassengers << endl;
    cout << "hours of operation: " << plane.hoursOperated << endl;
}