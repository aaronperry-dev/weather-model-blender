"""Shared constants."""

MM_PER_INCH = 25.4

# Precipitation type codes used internally
TYPE_NONE = 0
TYPE_RAIN = 1
TYPE_SNOW = 2
TYPE_SLEET = 3   # ice pellets
TYPE_FZRA = 4    # freezing rain

# Snow-level sentinels (metres MSL).
# COLD: wet-bulb is below the threshold all the way to the surface -> every terrain height is "above" it.
# WARM: no sub-threshold air in the column -> no terrain height is above it.
SNOW_LEVEL_COLD = -500.0
SNOW_LEVEL_WARM = 5000.0

WB_SNOW_THRESHOLD_C = 0.5     # WPC: snow level = height of the 0.5 degC wet-bulb surface
