
def fortran_float(value: float) -> str:
    '''Convert float to Fortran scientific notation with D instead of E.'''
    return (
        f"{value:.0E}"
        .replace("E", "D")
        .replace("-0", "-")
        .replace("+0", "+")
    )