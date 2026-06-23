import os
import json
import xml.etree.ElementTree as ET

def build_list_from_xml(xml_file, image_src_dir, fraction):
    '''Parses the XML file and builds a list of dictionaries containing image paths, bounding boxes, and labels.'''

    data_list = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    sequence_name = root.get('name')
    frames = root.findall('frame')
    # sample every nth frame
    step = int(1 / fraction)
    selected_frames = frames[::step]

    for frame in selected_frames:
        frame_number = frame.get('num')
        image_id = frame_number.zfill(5)
        image_path = os.path.join(image_src_dir, sequence_name, f"img{image_id}.jpg")
        target_list = frame.find('target_list')
        targets = []

        for target in target_list.findall('target'):
            bbox = target.find('box')
            box_left = float(bbox.get('left'))
            box_top = float(bbox.get('top'))
            box_width = float(bbox.get('width'))
            box_height = float(bbox.get('height'))
            targets.append([box_left, box_top, box_left + box_width, box_top + box_height])

        frame_dict = {
            "image_path": image_path,
            "boxes": targets,
            "labels": [1] * len(targets)  # Assuming all targets are of the same class (vehicle)
        }
        data_list.append(frame_dict)

    return data_list


if __name__ == "__main__":

    data_dir = "data"
    xml_dirs = ["DETRAC-Train-Annotations-XML", "DETRAC-Test-Annotations-XML"]
    image_src_dir = os.path.join(data_dir, "DETRAC-Images")
    frame_fraction = 0.1  # Use only 10% of each sequence's frames by sampling every 10th frame

    # go through train and test dirs
    for xml_dir in xml_dirs:
        all_data = []
        xml_path = os.path.join(data_dir, xml_dir)

        # go through each image sequence's xml file
        for xml_file in os.listdir(xml_path):
            if xml_file.endswith(".xml"):
                xml_file_path = os.path.join(xml_path, xml_file)
                data_list = build_list_from_xml(xml_file_path, image_src_dir, frame_fraction)
                all_data.extend(data_list)

        split_name = xml_dir.split('-')[1].lower()  # Extract 'train' or 'test' from the directory name
        output_file = os.path.join(data_dir, f"{split_name}.json")
        # create json for train and test data
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=4)