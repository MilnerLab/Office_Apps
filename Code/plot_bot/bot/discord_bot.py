# bot/discord_bot.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands, tasks

from Lab_apps.plot_bot.domain.config import BotConfig
from Lab_apps.plot_bot.domain.pipeline import process_scan_file
from Lab_apps.plot_bot.monitoring.scan_detector import DirectoryWatcher



class ScanDiscordBot(commands.Bot):
    """
    Discord bot that periodically asks a DirectoryWatcher for finished files
    and posts their plots (and optional sidecar text) into a channel.
    """

    def __init__(
        self,
        watcher: DirectoryWatcher,
        config: BotConfig
    ) -> None:
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)

        self.watcher = watcher
        self.channel_id = config.DISCORD_CHANNEL_ID
        self.check_interval = config.CHECK_INTERVAL

        self.target_channel: Optional[discord.TextChannel] = None

    async def setup_hook(self) -> None:
        """
        Called when the bot is initialized.
        Here we start the background scan loop with the desired interval.
        """
        self.scan_loop.change_interval(seconds=self.check_interval)
        self.scan_loop.start()

    async def on_ready(self) -> None:
        print(f"Logged in as {self.user} (ID: {self.user.id})")

        channel = self.get_channel(self.channel_id)
        if isinstance(channel, discord.TextChannel):
            self.target_channel = channel
            print(f"Using channel: #{channel.name} ({channel.id})")
        else:
            print(
                f"Warning: channel with ID {self.channel_id} not found. "
                "Check that the ID is correct and the bot is in the server."
            )

    # ------------------------------------------------------------------ #
    # Sidecar .txt handling
    # ------------------------------------------------------------------ #
    def get_sidecar_text(self, file_path: Path) -> Optional[str]:
        """
        Look for a text file with the same base name as the data file,
        but with extension .txt or .TXT, e.g.:

            20251113183351_ScanFile.dat
            20251113183351_ScanFile.txt

        Return its content as a string, or None if not found.
        """
        candidates = [
            file_path.with_suffix(".txt"),
            file_path.with_suffix(".TXT"),
        ]

        for txt_path in candidates:
            if txt_path.exists():
                try:
                    text = txt_path.read_text(encoding="utf-8", errors="replace")
                    return text
                except Exception as exc:
                    print(f"Error reading {txt_path}: {exc}")
                    return None

        return None

    def build_message_content(self, file_path: Path, sidecar_text: Optional[str]) -> str:
        """
        Build the Discord message content.

        - always: basic info about the finished scan
        - optional: content of the .txt file as a code block
        """
        base = f"New scan finished: `{file_path.name}`"

        if not sidecar_text:
            return base

        body = sidecar_text.strip()
        max_len = 1800  # Discord message limit is 2000 chars
        if len(body) > max_len:
            body = body[:max_len] + "\n...[truncated]"

        return base + "\n\n```text\n" + body + "\n```"

    # ------------------------------------------------------------------ #
    # Background loop
    # ------------------------------------------------------------------ #
    @tasks.loop(seconds=60.0)  # interval overridden in setup_hook
    async def scan_loop(self) -> None:
        if self.target_channel is None:
            return

        finished_files = self.watcher.scan()
        for file_path in finished_files:
            await self.handle_finished_file(file_path)

    @scan_loop.before_loop
    async def before_scan_loop(self) -> None:
        await self.wait_until_ready()
        print("Scan loop started.")

    # ------------------------------------------------------------------ #
    # Handling a finished file
    # ------------------------------------------------------------------ #
    async def handle_finished_file(self, file_path: Path) -> None:
        """
        Called when the watcher reports a finished .dat file.
        """
        print(f"[Watcher] Finished file detected: {file_path}")

        # 1) Create plot
        try:
            png_path = process_scan_file(file_path)
        except Exception as exc:
            msg = f"Error while processing `{file_path.name}`: `{exc}`"
            print(msg)
            await self.target_channel.send(msg)
            return

        # 2) Optional .txt sidecar
        sidecar_text = self.get_sidecar_text(file_path)

        # 3) Build message content
        content = self.build_message_content(file_path, sidecar_text)

        # 4) Send to Discord
        try:
            discord_file = discord.File(str(png_path), filename=png_path.name)
            await self.target_channel.send(content=content, file=discord_file)
            print(f"[Watcher] Plot sent: {png_path}")
        except Exception as exc:
            print(f"Error sending message to Discord: {exc}")

    async def on_command_error(self, ctx: commands.Context, error: Exception) -> None:
        await ctx.send(f"Error: {error}")
