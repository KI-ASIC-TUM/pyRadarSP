#!/usr/bin/env python3
"""
Download 6 representative scenarios

* new_1: One pedestrian and one cyclist moving next to a static car.
Ego-car takes over the three of them
* new_2: Overtaking of a pedestrian on a clear environment
* new_3: Ego-car approaches a crossing and stops behind another car
* new_4: Ego-car follows a second car through a narrow street
* new_5: Ego-car moves turns in a crossing with low noise around
* new_6: Ego-car moves through a crowded road with multiple moving cars
* new_7: Ego-car in countryside crosses a car at fast speed
"""
# Standard libraries
import numpy as np
# Local libraries
import dhandler.oth_handler


def main():
    SCENES = {
        "new_1": "20230127_114455",
        "new_2": "20230125_131034",
        "new_3": "20230314_132412",
        "new_4": "20230314_133320",
        "new_5": "20230314_132605",
        "new_6": "20230314_132927",
        "new_7": "20230127_092527",
    }
    handler = dhandler.oth_handler.OTHHandler("OTH/20230127_114455")
    for scene in SCENES:
        handler.dataset_name = "OTH/"+SCENES[scene]
        handler.load(skip_permission=True)


if __name__=="__main__":
    main()
