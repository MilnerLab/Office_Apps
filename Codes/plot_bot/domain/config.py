from dataclasses import dataclass

from Lab_apps._io.dat_finder import MOST_RECENT_FOLDER

@dataclass
class BotConfig:
    WATCH_DIR = MOST_RECENT_FOLDER

    CHECK_INTERVAL = 5.0
    INACTIVITY_THRESHOLD = 250
    DYNAMIC_INACTIVITY_MULTIPLIER = 3

    DISCORD_TOKEN = "MTQ0MDEwMjQxMDgyNjY4MjQyMA.GeZCGk.qzl2pAn_GfyXn9seyE57-O510IGjTZe9N8ZIxo"
    DISCORD_CHANNEL_ID = 1440145408620888114  
