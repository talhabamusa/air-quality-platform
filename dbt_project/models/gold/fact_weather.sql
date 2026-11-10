select
    md5(w.city || '|' || (w.hour_utc - interval '30 minutes')::text) as weather_key,
    w.hour_utc - interval '30 minutes' as hour_key,
    w.hour_utc as weather_hour_utc,
    w.city as city_key,
    w.temp_c,
    w.humidity_pct,
    w.pressure_hpa,
    w.precip_mm,
    w.wind_ms,
    w.wind_dir_deg
from {{ source('staging', 'stg_weather_hourly') }} as w