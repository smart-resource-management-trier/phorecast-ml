import numpy
import pandas
import pvlib


def attach_solar_positions(
        data: pandas.DataFrame, latitude: float, longitude: float, height: float = 0, solar_parameters:list[str] = ["azimuth", "elevation", "zenith"]
) -> pandas.DataFrame:
    """
    Attaches the solar positions to the given data
    :param data: dataframe to take timestamps from and attach solar positions to
    :param latitude: latitude of the location
    :param longitude: longitude of the location
    :param height: height of the location
    :param solar_parameters:

    :return: dataframe with solar positions attached
    """

    data = data.copy()
    if data.columns.intersection(solar_parameters).size > 0:
        raise ValueError(f"Data probably already contains following solar positions: {data.columns.intersection(solar_parameters)}")

    loc = pvlib.location.Location(latitude=latitude, longitude=longitude, altitude=height)
    times = data.index
    solar_position = loc.get_solarposition(times)[solar_parameters]
    solar_position["clear_sky_GHI"] = loc.get_clearsky(times)["ghi"]

    data = pandas.concat([data, solar_position], axis=1)

    return data

def attach_cyclic_data(data: pandas.DataFrame):
    data = data.copy()
    day_of_year = data.index.dayofyear

    data["sin_doy"] = numpy.sin(2 * numpy.pi * (day_of_year - 1) / 365)
    data["cos_doy"] = numpy.cos(2 * numpy.pi * (day_of_year - 1) / 365)

    seconds_in_day = (
            data.index.hour * 3600 +
            data.index.minute * 60 +
            data.index.second
    )

    data["sin_time"] = numpy.sin(2 * numpy.pi * seconds_in_day / 86400)
    data["cos_time"] = numpy.cos(2 * numpy.pi * seconds_in_day / 86400)

    return data