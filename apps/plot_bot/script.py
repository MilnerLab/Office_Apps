# main.py

from _data_io.dat_finder import SCAN_FILE_PATTERN
from apps.plot_bot.domain.config import BotConfig
from apps.plot_bot.monitoring.scan_detector import DirectoryWatcher
from apps.plot_bot.bot.discord_bot import ScanDiscordBot


def main() -> None:

    config = BotConfig()

    if not config.WATCH_DIR.exists():
        raise RuntimeError(f"WATCH_DIR does not exist: {config.WATCH_DIR}")

    watcher = DirectoryWatcher(config)

    bot = ScanDiscordBot(watcher, config)

    print(f"Watching directory: {config.WATCH_DIR} (pattern: {SCAN_FILE_PATTERN})")
    print(
        f"Check interval: {config.CHECK_INTERVAL}s, "
        f"base inactivity threshold: {config.INACTIVITY_THRESHOLD}s, "
        f"dynamic factor: {config.DYNAMIC_INACTIVITY_MULTIPLIER}"
    )

    bot.run(config.DISCORD_TOKEN)


if __name__ == "__main__":
    main()
