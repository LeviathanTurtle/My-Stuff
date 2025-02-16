
#
# Simple unit converter
# WILLIAM WADSWORTH
# Usage: python -m -k unit_converter.py
# 

from unittest import TestCase
from unittest import main as unittest_main

LENGTH_CONVERSION_RATES = {
    "Meter": 1,           # base
    "Mile": 1609.344,     # Miles to Meters
    "Kilometer": 1000,    # Kilometers to Meters
    "Foot": 0.3048,       # Feet to Meters
    "Yard": 0.9144,       # Yards to Meters
    "Inch": 0.0254,       # Inches to Meters
    "Centimeter": 0.01,   # Centimeters to Meters
    "Millimeter": 0.001,  # Millimeters to Meters
    "Nanometer": 1e-9,    # Nanometers to Meters
}

ASTRONOMY_CONVERSION_RATES = {
    "Astronomical Unit": 1,  # base
    "Light Year": 63240.87,  # 1 light year = 63,240.87 AU
    "Parsec": 206266.3,      # 1 parsec = 206,266.3 AU
}

WEIGHT_CONVERSION_RATES = {
    "Gram": 1,           # base
    "Pound": 453.592,    # Pounds to grams
    "Kilogram": 1000,    # Kilograms to grams
    "Ounce": 28.3495,    # Ounces to grams
    "Metric ton": 1e6,   # Metric tons to grams
    "Stone": 6350.29,    # Stones to grams
    "Milligram": 0.001,  # Milligrams to grams
    "Microgram": 1e-6,   # Micrograms to grams
}

FORCE_CONVERSION_RATES = {
    "Newton": 1,                # base
    "Pound-force": 4.44822,     # Pounds-force to Newtons
    "Kilogram-force": 9.80665,  # Kilogram-force to Newtons
    "Dynes": 1e-5,              # Dynes to Newtons
    "Kilonewton": 1000,         # Kilonewtons to Newtons
    "Meganewton": 1e6,          # Meganewtons to Newtons
}

VOLUME_CONVERSION_RATES = {
    "Liter": 1,                     # base
    "Gallon (US)": 0.264172,        # Gallons (US) to Liters
    "Cubic meter": 1000,            # Cubic meters to Liters
    "Milliliter": 0.001,            # Milliliters to Liters
    "Pint (US)": 0.473176,          # Pints (US) to Liters
    "Quart (US)": 0.946353,         # Quarts (US) to Liters
    "Fluid ounce (US)": 0.0295735,  # Fluid ounces (US) to Liters
    "Cubic centimeter": 0.001,      # Cubic centimeters to Liters
    "Cubic inch": 0.0163871,        # Cubic inches to Liters
    "Cubic foot": 28.3168,          # Cubic feet to Liters
}

AREA_CONVERSION_RATES = {
    "Square meter": 1,            # base
    "Square mile": 2.58999e6,     # Square miles to Square meters
    "Square kilometer": 1e6,      # Square kilometers to Square meters
    "Square foot": 0.092903,      # Square feet to Square meters
    "Acre": 4046.86,              # Acres to Square meters
    "Hectare": 10000,             # Hectares to Square meters
    "Square centimeter": 0.0001,  # Square centimeters to Square meters
    "Square inch": 0.00064516,    # Square inches to Square meters
    "Square yard": 0.836127       # Square yards to Square meters
}

"""TEMPERATURE_CONVERSION_RATES = {
    ("Celsius", "Fahrenheit"): lambda c: (c * 9/5) + 32,
    ("Fahrenheit", "Celsius"): lambda f: (f - 32) * 5/9,
    ("Celsius", "Kelvin"): lambda c: c + 273.15,
    ("Kelvin", "Celsius"): lambda k: k - 273.15,
    ("Fahrenheit", "Kelvin"): lambda f: (f - 32) * 5/9 + 273.15,
    ("Kelvin", "Fahrenheit"): lambda k: (k - 273.15) * 9/5 + 32,
}"""

TIME_CONVERSION_RATES = {
    "Day": 1,           # base
    "Hour": 1/24,       # 1 day = 24 hours
    "Minute": 1/1440,   # 1 day = 1440 minutes (24*60)
    "Second": 1/86400,  # 1 day = 86400 seconds (24*60*60)
    "Week": 7,          # 1 week = 7 days
    "Month": 30.44,     # Average month = 30.44 days
    "Year": 365.24,     # Average year = 365.24 days
    "Decade": 3655.4,   # 1 decade = 10 years
}

VELOCITY_CONVERSION_RATES = {
    "Meters per second": 1,       # base
    "Miles per hour": 1/2.23694,  # 1 m/s = 2.23694 mph
    "Kilometers per hour": 3.6,   # 1 m/s = 3.6 km/h
    "Knots": 1/1.94384            # 1 m/s = 1.94384 knots
}

"""ELECTRICITY_CONVERSION_RATES = {
    # Coulombs to Amperes (current over time)
    ("Coulombs", "Amperes (per second)"): 1,  # 1 C = 1 A·s
    ("Amperes (per second)", "Coulombs"): 1,
    # Ohm's Law: Voltage = Current * Resistance (V = I * R)
    ("Volts", "Amperes (given Ohms)"): lambda V, R: V / R,  # V = I * R -> I = V / R
    ("Amperes", "Volts (given Ohms)"): lambda I, R: I * R,  # I = V / R -> V = I * R
    ("Ohms", "Amperes (given Volts)"): lambda R, V: V / R,  # R = V / I -> I = V / R
    # Power Law: Power = Voltage * Current (P = V * I)
    ("Watts", "Volts (given Amperes)"): lambda P, I: P / I,  # P = V * I -> V = P / I
    ("Volts", "Watts (given Amperes)"): lambda V, I: V * I,  # V = P / I -> P = V * I
    ("Watts", "Amperes (given Volts)"): lambda P, V: P / V,  # P = V * I -> I = P / V
    # Resistance: Ohms
    ("Ohms", "Volts (given Amperes)"): lambda R, I: I * R,  # Ohms law: V = I * R
}
def convert_electricity(value, from_unit, to_unit, *args):
    if isinstance(ELECTRICITY_CONVERSION_RATES[(from_unit, to_unit)], float):
        # Simple numeric conversion
        return value * ELECTRICITY_CONVERSION_RATES[(from_unit, to_unit)]
    else:
        # Handle complex conversions (Ohm's Law or Power Law)
        return ELECTRICITY_CONVERSION_RATES[(from_unit, to_unit)](value, *args)
# Example: Convert 10 Volts with 2 Ohms to Amperes
amps = convert_electricity(10, "Volts", "Amperes (given Ohms)", 2)
print(amps)  # Output: 5.0 (because 10V / 2Ω = 5A)"""

FREQUENCY_CONVERSION_RATES = {
    "Hertz": 1,              # base
    "Cycles per second": 1,  # 
    "Kilohertz": 1000,       # 1 Hz = 0.001 kHz
    "Megahertz": 1e6,        # 1 Hz = 0.000001 MHz
    "Gigahertz": 1e9,        # 1 Hz = 0.000000001 GHz
}

PRESSURE_CONVERSION_RATES = {
    "Pascals": 1,           # base
    "Bar": 1e5,             # 1 bar = 100,000 Pascals
    "Atmospheres": 101325,  # 1 atm = 101,325 Pascals
    "PSI": 6894.76,         # 1 PSI = 6,894.76 Pascals
    "Torr": 133.322,        # 1 torr = 133.322 Pascals
}

ENERGY_CONVERSION_RATES = {
    "Joules": 1,              # base
    "Calories": 4.184,        # 1 cal = 4.184 Joules
    "Kilowatt-hours": 3.6e6,  # 1 kWh = 3,600,000 Joules
    "Kilocalories": 4184,     # 1 kcal = 4184 Joules
    "Watt-hours": 3600,       # 1 Wh = 3600 Joules
}

POWER_CONVERSION_RATES = {
    "Watts": 1,              # base
    "Kilowatts": 1000,       # 1 kW = 1000 W
    "Megawatts": 1_000_000,  # 1 MW = 1,000,000 W
    "Horsepower": 745.701,   # 1 hp ~ 745.701 W
}

DATA_CONVERSION_RATES = {
    "Byte": 1,                      # base
    "Kilobyte": 1000,               # 1 KB = 1000 Bytes
    "Kibibyte": 1024,               # 1 KiB = 1024 Bytes
    "Megabyte": 1_000_000,          # 1 MB = 1,000,000 Bytes
    "Mebibyte": 1_048_576,          # 1 MiB = 1,048,576 Bytes
    "Gigabyte": 1_000_000_000,      # 1 GB = 1,000,000,000 Bytes
    "Gibibyte": 1_073_741_824,      # 1 GiB = 1,073,741,824 Bytes
    "Terabyte": 1_000_000_000_000,  # 1 TB = 1,000,000,000,000 Bytes
    "Tibibyte": 1_099_511_627_776,  # 1 TiB = 1,099,511,627,776 Bytes
}

CONVERSION_RATES = {
    "Length": LENGTH_CONVERSION_RATES,
    "Astronomy": ASTRONOMY_CONVERSION_RATES,
    "Weight": WEIGHT_CONVERSION_RATES,
    "Force": FORCE_CONVERSION_RATES,
    "Volume": VOLUME_CONVERSION_RATES,
    "Area": AREA_CONVERSION_RATES,
    #"Temperature": TEMPERATURE_CONVERSION_RATES,
    "Time": TIME_CONVERSION_RATES,
    "Velocity": VELOCITY_CONVERSION_RATES,
    #"Electricity": ELECTRICITY_CONVERSION_RATES,
    "Frequency": FREQUENCY_CONVERSION_RATES,
    "Pressure": PRESSURE_CONVERSION_RATES,
    "Energy": ENERGY_CONVERSION_RATES,
    "Power": POWER_CONVERSION_RATES,
    "Data": DATA_CONVERSION_RATES,
}

def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Converts a value from one unit to another."""
    
    # Find the category of conversion
    from_category = None
    to_category = None

    # Identify the categories for both units
    for category, rates in CONVERSION_RATES.items():
        if from_unit in rates: from_category = category
        if to_unit in rates: to_category = category

    # If either unit is not found or they are in different categories
    if from_category is None or to_category is None:
        raise ValueError(f"Conversion from '{from_unit}' to '{to_unit}' is not supported.")
    if from_category != to_category:
        raise ValueError(f"Cannot convert between different categories: '{from_category}' and '{to_category}'.")

    base_value = value * CONVERSION_RATES[from_category][from_unit] # convert to base unit of the dict
    return base_value/CONVERSION_RATES[to_category][to_unit] # convert from base to target unit

class TestConversions(TestCase):
    def test_all_conversions(self):
        for category, rates in CONVERSION_RATES.items():
            units = list(rates.keys())
            for from_unit in units:
                for to_unit in units:
                    with self.subTest(from_unit=from_unit, to_unit=to_unit, category=category):
                        # Test conversion from base unit to itself
                        if from_unit == to_unit:
                            self.assertAlmostEqual(
                                convert(1, from_unit, to_unit),
                                1,
                                places=6,
                                msg=f"Conversion from {from_unit} to {to_unit} in {category} should be 1"
                            )
                        else:
                            # Test converting 1 unit to another and back
                            value = 1.0
                            converted = convert(value, from_unit, to_unit)
                            reverted = convert(converted, to_unit, from_unit)
                            # Allow a small tolerance for floating-point arithmetic
                            self.assertAlmostEqual(
                                reverted, value, places=6,
                                msg=f"Conversion from {from_unit} to {to_unit} and back in {category} should yield the original value"
                            )

if __name__ == "__main__":
    unittest_main()