# Team: Foobar
# Teammates: Anthony Cox, Corey Lawrence, Dylan Hudson, Parker Blue, Will Wadsworth, Zach Christopher
# Authors: Corey Lawrence, Dylan Hudson, Will Wadsworth
# Date: 
#
# Description:
#   This module defines and implements the model class `Airport`.

import decimal
from queue import Queue
from typing import Type
from aircraft import AircraftType

class Airport:
    """Model class. A generic representation of an airport."""
    def __init__(
            self, name: str, iata_code: str, city: str, state: str, metro_population: int, is_hub: bool,
            available_gates: int, latitude: float, longitude: float, gas_price: decimal, takeoff_fee: decimal,
            landing_fee: decimal, tarmac: Queue[Type[AircraftType]]#, paris_connected: bool
        ):
        self.name = name
        self.iata_code = iata_code
        self.city = city
        self.state = state
        self.metro_population = metro_population
        self.is_hub = is_hub
        self.available_gates = available_gates
        self.latitude = latitude
        self.longitude = longitude
        self.gas_price = gas_price
        self.takeoff_fee = takeoff_fee
        self.landing_fee = landing_fee
        self.tarmac = tarmac # ?
        #self.paris_connected = paris_connected