
#include <iostream>
using namespace std;

// 10 locations
string hubs[] = {"Atlanta","Dallas","Denver","Chicago","Los Angeles","New York City","Las Vegas","Orlando","Miami","Charlotte"};
// gate letter possibilities -- 7
char gateLetters[] = {'A','B','C','D','E','F','G'};
// gate number possibilities -- 10
int gateNumbers[] = {1,2,3,4,5,6,7,8,9,10};



struct location {
    int degree;
    char direction;
};

struct gate {
    char gateChar = gateLetters[0];
    int gateNum = gateNumbers[0];
}

struct planeObj {
    string departDestination = hubs[0];
    gate departGate;
    string arriveDestination = hubs[1];
    gate arriveGate;
    int numPassengers;
    int hoursOperated;
    double percentOnTimeArrival;
    double percentOnTimeDeparture;
    char flightPath;

    // keys
    int tailNum;
    int flightNum;

    // location
    location locationOfPlane;
};

struct airport {
    string name;
    location locationOfAirport;
}

// global total flights
int totalFlights;
// global report vars
int globalPassengersTransported;
double globalOperatingCost;
double globalRevenue;
double globalProfitOrLoss;

// GET METRO AREA AROUND AIRPORT FOR PASSENGERS


int main()
{
    planeObj plane;

    outputPlaneInfo(plane);

    // repeat for each day
    for(int day=1; day<=10; day++) {
        // check hindrances
        hindrances(day, plane);

        // LOADING
        // update gate

        // update arrive destination
        // update depart destination
        // update number of passengers
        // start timer

        // LAND
        // stop timer
        // update flight time
        // (if applicable): update percent on time arrival/departure
        // increment hours operated

        // UPDATE REPORT
        globalPassengersTransported += plane.numPassengers;
        //globalOperatingCost += plane.hoursOperated * // something
        //globalRevenue += numPassengers * // ticket price
        //globalProfitOrLoss += // operating cost +/- revenue
        //    if(operatingCost - revenue < 0) ? cout << "loss of " << abs(operatingCost - revenue) << endl : cout << "gain of " << operatingCost - revenue << endl;

        outputPlaneInfo(plane);
        cout << "----- END OF DAY -----" << endl << endl << endl;
    }
    // ^ repeat for each day

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
    cout << "degree of location: " << plane.location.degree << endl;
    cout << "direction of location: " << plane.location.direction << endl;

    cout << "arriving at: " << plane.arriveDestination << endl;
    cout << "flight path: " << plane.flightPath << endl;
    cout << "number of passengers on this flight: " << plane.numPassengers << endl;
    cout << "hours of operation: " << plane.hoursOperated << endl;
}