from python_scripts.process_real_data.raw_data_new_format.parse_xml_file import parse_xml_file
from shapely import geometry
import numpy as np
import csv


def generate_sensor_locations_file_from_poly(poly: geometry.polygon = None):
    """

    :param poly:
    :return:
    """
    stations_dict = parse_xml_file("python_scripts/process_real_data/raw_data_new_format/xml_file.xml")
    print(stations_dict)

    sensors_lat_lon_dict = {}

    for station_id in stations_dict:
        lat, lon = float(stations_dict[station_id]["lat"]), float(stations_dict[station_id]["lon"])

        # choose a random direction; out of two directions NE/SW or NW/SE
        # this is just a smart quickfix, ideally this should be perpendicular to the road direction
        if np.random.rand() < 0.5:
            dir = [+1, -1]
        else:
            dir = [+1, +1]

        how_many_sensors = len(stations_dict[station_id]["detectors"])

        # choose how_many_sensor locations in the direction chosen, around the home station lat,lon
        # but how far apart should they be?
        # we find the distance between the opposite ends of a road in this neighbourhood and use its
        # difference of lat, lon as our guide
        end_1 = 44.92536439315752, -93.27446100020875
        end_2 = 44.92537579315532, -93.27455890082871

        # net changes:
        lat_diff = abs(end_1[0] - end_2[0])
        lon_diff = abs(end_1[1] - end_2[1])

        for j in range(how_many_sensors):
            sensors_lat_lon_dict[stations_dict[station_id]["detectors"][j]] = {
                "lat": lat + dir[0] * lat_diff / how_many_sensors * (j + 1),
                "lon": lon + dir[1] * lon_diff / how_many_sensors * (j + 1),
            }

    # filter stations and sensors inside poly
    if poly is not None:

        # .copy() use to avoid the error: RuntimeError: dictionary changed size during iteration
        for sensor in sensors_lat_lon_dict.copy():
            sensor_point = geometry.Point(sensors_lat_lon_dict[sensor]["lon"], sensors_lat_lon_dict[sensor]["lat"])
            if not poly.contains(sensor_point):
                del sensors_lat_lon_dict[sensor]

        for station in stations_dict.copy():
            station_point = geometry.Point(float(stations_dict[station]["lon"]), float(stations_dict[station]["lat"]))
            if not poly.contains(station_point):
                del stations_dict[station]

    # generate kepler file for the sensor locations
    with open("output_images/kepler_files/sensor_ids_detectors.csv", "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["lat", "lon", "sensor_id"])
        for sensor in sensors_lat_lon_dict:
            csvwriter.writerow([sensors_lat_lon_dict[sensor]["lat"], sensors_lat_lon_dict[sensor]["lon"], sensor])

    # generate kepler file for the sensor locations
    with open("output_images/kepler_files/station_ids.csv", "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["lat", "lon", "station_id"])
        for station in stations_dict:
            csvwriter.writerow([stations_dict[station]["lat"], stations_dict[station]["lon"], station])


if __name__ == "__main__":
    geo = {
        "type": "Polygon",
        "coordinates": [
            [
                [-93.27890396118164, 44.968441538936894],
                [-93.28353881835938, 44.96018236052143],
                [-93.27881813049316, 44.90972091446053],
                [-93.27098608016968, 44.90934099435474],
                [-93.26079368591309, 44.95793517240357],
                [-93.2603645324707, 44.9671663022509],
                [-93.26911926269531, 44.970870482764084],
                [-93.27890396118164, 44.968441538936894],
            ]
        ],
    }
    poly = geometry.Polygon([tuple(l) for l in geo["coordinates"][0]])

    generate_sensor_locations_file_from_poly(poly)
