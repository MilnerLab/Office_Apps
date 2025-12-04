from dataclasses import dataclass
import os

from Lab_apps._io.dat_finder import MOST_RECENT_FOLDER

@dataclass
class BotConfig:
    WATCH_DIR = MOST_RECENT_FOLDER

    CHECK_INTERVAL = 5.0
    INACTIVITY_THRESHOLD = 250
    DYNAMIC_INACTIVITY_MULTIPLIER = 3

    
    # MTQ0NTkwNjE2Mjg3NTc2MDc5Mw. GRgFDk. utLEQVHiwHTZONkCTbnJv7lytkoWqqZGnlWick   <- remove spaces
    # $env:DISCORD_TOKEN = "BOT_TOKEN"
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    DISCORD_CHANNEL_ID = 1440145408620888114  
