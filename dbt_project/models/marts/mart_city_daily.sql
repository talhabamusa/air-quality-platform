with pm as (
    select
        f.city_key,
        f.sensor_id,
        h.date_local,
        f.pm25_ugm3
    from {{ ref('fact_air_quality') }} as f
    inner join {{ ref('dim_hour') }} as h
        on h.hour_key = f.hour_key
    where f.quality_tier in ('observed', 'imputed')
),

sensor_day as (
    select
        city_key,
        sensor_id,
        date_local,
        count(*) as valid_hours,
        avg(pm25_ugm3) as pm25_mean
    from pm
    group by 1, 2, 3
    having count(*) >= {{ var('min_valid_hours_per_sensor_day', 18) }}
),

city_day as (
    select
        city_key,
        date_local,
        count(*) as n_sensors,
        avg(pm25_mean) as pm25_mean,
        percentile_cont(0.5) within group (order by pm25_mean) as pm25_median
    from sensor_day
    group by 1, 2
    having count(*) >= {{ var('min_sensors_per_city_day', 3) }}
),

weather_day as (
    select
        w.city_key,
        h.date_local,
        count(*) as weather_hours,
        avg(w.temp_c) as temp_mean_c,
        avg(w.humidity_pct) as humidity_mean_pct,
        sum(w.precip_mm) as precip_total_mm,
        avg(w.wind_ms) as wind_mean_ms
    from {{ ref('fact_weather') }} as w
    inner join {{ ref('dim_hour') }} as h
        on h.hour_key = w.hour_key
    group by 1, 2
)

select
    c.city_key,
    c.date_local,
    c.n_sensors,
    round(c.pm25_mean::numeric, 1) as pm25_mean_ugm3,
    round(c.pm25_median::numeric, 1) as pm25_median_ugm3,
    wd.weather_hours,
    round(wd.temp_mean_c::numeric, 1) as temp_mean_c,
    round(wd.humidity_mean_pct::numeric, 1) as humidity_mean_pct,
    round(wd.precip_total_mm::numeric, 1) as precip_total_mm,
    round(wd.wind_mean_ms::numeric, 1) as wind_mean_ms
from city_day as c
left join weather_day as wd
    on wd.city_key = c.city_key
   and wd.date_local = c.date_local