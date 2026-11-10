select
    city as city_key,
    display_name as city_name,
    centre_lat,
    centre_lon,
    weather_grid_lat,
    weather_grid_lon
from {{ ref('seed_cities') }}