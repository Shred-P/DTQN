from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.core.util import deep_dict_merge
from mobile_env.core.movement import ArticleRandomWaypointMovement
import numpy as np

def create_fixed_medium_map(bs_dist=100, dist_to_border=50):
    """
    Create  3 base stations (BS) positioned at equal distances in an equilateral triangle layout.
    """
    # calculate vertical distance from A, B to C using Pythagoras for an equilateral triangle layout
    y_dist = np.sqrt(bs_dist ** 2 - (bs_dist / 2) ** 2)

    station_pos = [(dist_to_border,dist_to_border),(dist_to_border + bs_dist,dist_to_border),(dist_to_border + (bs_dist / 2),dist_to_border + y_dist)]
    # Set up BS positions based on the map size and borders

    return station_pos

class MComArticle(MComCore):
    def __init__(self, config={}, render_mode=None):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)

        config['utility_params']['lower'] = -10
        config['utility_params']['upper'] = 10

        config['movement'] = ArticleRandomWaypointMovement # 1到3米的均匀分布

        station_pos = create_fixed_medium_map(bs_dist=config['bs_dist'], dist_to_border=50)
        # station_pos = [(110, 130), (65, 80), (120, 30)]
        stations = [
            BaseStation(bs_id, pos, **config["bs"])
            for bs_id, pos in enumerate(station_pos)
        ]
        num_ues = config['num_ues']
        ues = [UserEquipment(ue_id, **config["ue"]) for ue_id in range(num_ues)]

        super().__init__(stations, ues, config, render_mode)
