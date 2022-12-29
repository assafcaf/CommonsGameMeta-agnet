from Danfoa.envs.commons_env import HarvestCommonsEnv as HarvestEnv
from Danfoa.envs.commons_meta_env import MetaHarvestCommonsEnv

from Danfoa.maps import *


def get_env_creator(
    env,
    num_agents,
    k=25,
    harvest_view_size=7,
    map_env=HARVEST_MAP,
    ep_length=1000,
):
    if env == "Harvest":

        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                harvest_view_size=harvest_view_size,
                ascii_map=map_env,
                ep_length=ep_length
            )
    elif env == "HarvestMeta":
        def env_creator(_):
            return MetaHarvestCommonsEnv(
                num_agents=num_agents,
                harvest_view_size=harvest_view_size,
                ascii_map=map_env,
                ep_length=ep_length,
                k=k
            )
    else:
        raise ValueError(f"env must be one of Harvest or HarvestMeta, not {env}")

    return env_creator
