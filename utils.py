import re

def get_digit_and_unit(value):

    unit_rex = r"(km|ml|mm|mg|kg|cm|m|g|l)"
    digit_rex = r"(\d+\.?\d*)"
    
    units = re.findall(unit_rex, value)
    digits = re.findall(digit_rex, value)
    
    digits = float(digits[0]) if len(digits) > 0 else 0
    units = units[0] if len(units) > 0 else None
    return digits, units