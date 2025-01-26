import xml.etree.ElementTree as ET
from copy import deepcopy


def parse_xml_file(input_file_name):
    stations_dict = {}
    tree = ET.parse(input_file_name)
    root = tree.getroot()
    for child in root:
        for child_2 in child:
            if "station_id" in child_2.attrib:
                temp = {}
                temp["station_id"] = child_2.attrib["station_id"]
                temp["lat"] = child_2.attrib["lat"]
                temp["lon"] = child_2.attrib["lon"]
                temp["detectors"] = []
                for detector in child_2:
                    temp["detectors"].append(detector.attrib["name"])

                stations_dict[child_2.attrib["station_id"]] = deepcopy(temp)
    return stations_dict


if __name__ == "__main__":
    stations_dict = parse_xml_file("xml_file.xml")

    for k, v in list(stations_dict.items())[:5]:
        print(k, v)
