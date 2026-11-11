select *
from {{ ref('mart_city_daily') }}
where n_sensors < {{ var('min_sensors_per_city_day', 3) }}