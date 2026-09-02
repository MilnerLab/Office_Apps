from dataclasses import dataclass
import os

from _data_io.dat_finder import MOST_RECENT_FOLDER


@dataclass
class BotConfig:
    WATCH_DIR = MOST_RECENT_FOLDER

    CHECK_INTERVAL = 5.0
    INACTIVITY_THRESHOLD = 100
    DYNAMIC_INACTIVITY_MULTIPLIER = 3


    # Set the token in the environment before starting the bot, e.g.
    #   $env:DISCORD_TOKEN = "<bot token>"
    # Never commit the token itself.
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    DISCORD_CHANNEL_ID = 1440145408620888114  
