with active_sensors as (
    select sensor_id from {{ source('staging', 'stg_pm25_hourly') }}
    union
    select sensor_id from {{ source('staging', 'stg_quarantine') }}
)

select
    s.location_id as station_key,
    s.location_name as station_name,
    s.city as city_key,
    s.sensor_id
from {{ source('staging', 'stg_sensors') }} as s
inner join active_sensors as a
    on a.sensor_id = s.sensor_id