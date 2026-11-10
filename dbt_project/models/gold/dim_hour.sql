with bounds as (
    select
        min(hour_utc) - interval '30 minutes' as first_hour,
        max(hour_utc) - interval '30 minutes' as last_hour
    from {{ source('staging', 'stg_weather_hourly') }}
),

hours as (
    select generate_series(first_hour, last_hour, interval '1 hour') as hour_key
    from bounds
),

local_time as (
    select
        hour_key,
        hour_key at time zone 'Asia/Kolkata' as hour_start_local
    from hours
)

select
    hour_key,
    hour_start_local,
    hour_start_local::date as date_local,
    extract(hour from hour_start_local)::int as hour_of_day_local,
    extract(isodow from hour_start_local)::int as iso_weekday,
    trim(to_char(hour_start_local, 'Day')) as weekday_name,
    extract(isodow from hour_start_local) in (6, 7) as is_weekend
from local_time