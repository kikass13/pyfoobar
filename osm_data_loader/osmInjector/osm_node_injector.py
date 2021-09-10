from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET
import yaml


with open("parking_spots.yaml", 'r') as stream:
    try:
        data_loaded=yaml.safe_load(stream)
        parking_spaces = data_loaded['parking_spaces']
    except yaml.YAMLError as exc:
        print(exc)


tree = ET.parse('map.osm')
root = tree.getroot()

for i, spot in enumerate(parking_spaces):

    parking_spot_name_list = list(parking_spaces.keys())
    parking_space_id = parking_spot_name_list[i][1:]

    if (len(parking_space_id)>3):
        parking_space_id = parking_space_id[:3] + '1'

    print(parking_space_id)

    new_node = ET.Element("node")
    new_node.set('id',parking_space_id)
    new_node.set('visible','true')
    new_node.set('version', '2')
    new_node.set('changeset', '50302094')
    new_node.set('timestamp','2017-07-15T09:40:10Z')
    new_node.set('user', 'user')
    new_node.set('uid', '1776033')
    new_node.set('lat', str(parking_spaces[spot]['latitude']))
    new_node.set('lon', str(parking_spaces[spot]['longitude']))

    tag1 = SubElement(new_node,'tag')
    tag1.set('k', 'parking_spot')
    tag1.set('v', 'yes')

    tag2 = SubElement(new_node, 'tag')
    tag2.set('k', 'name')
    tag2.set('v' , spot)

    root.append(new_node)
    
    
tree.write('pz_parking_spots.osm')