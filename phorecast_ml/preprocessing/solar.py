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

    if data.columns.intersection(solar_parameters).size > 0:
        raise ValueError(f"Data probably already contains following solar positions: {data.columns.intersection(solar_parameters)}")

    loc = pvlib.location.Location(latitude=latitude, longitude=longitude, altitude=height)
    times = data.index
    solar_position = loc.get_solarposition(times)[solar_parameters]

    data = pandas.concat([data, solar_position], axis=1)

    return data