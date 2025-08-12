import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict
import re
import cv2
import yt_dlp

# NOTE: This file no longer uses or requires the youtube-transcript-api library.

@dataclass
class VideoMetadata:
    title: str
    author: str
    views: int
    duration: int
    filesize_mb: float
    format: str
    resolution: str
    video_id: str

class VideoProcessor:
    def __init__(self, url: str, config: dict):
        self.url = url
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.video_id = self._extract_video_id(url)
        self._progress_callback = None

    def _extract_video_id(self, url: str) -> str:
        try:
            if "?v=" in url:
                return url.split("?v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]
            raise ValueError("Invalid YouTube URL format")
        except Exception as e:
            self.logger.error(f"Failed to extract video ID from URL: {url} - {e}")
            raise ValueError(f"Invalid YouTube URL: {e}")

    def get_video_info(self, progress_callback=None) -> Tuple[VideoMetadata, str]:
        self._progress_callback = progress_callback
        ydl_opts = {
            "format": "best",
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [self._progress_hook],
        }

        try:
            self.logger.info(f"Fetching video info for: {self.url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)

            metadata = VideoMetadata(
                title=info.get("title", "Unknown"),
                author=info.get("uploader", "Unknown"),
                views=info.get("view_count", 0),
                duration=int(info.get("duration", 0)),
                filesize_mb=0,
                format=info.get("format", "Unknown"),
                resolution=f"{info.get('width', 'Unknown')}x{info.get('height', 'Unknown')}",
                video_id=self.video_id,
            )
            stream_url = info['url']
            self.logger.info("Successfully fetched video info and stream URL.")
            return metadata, stream_url
        except Exception as e:
            self.logger.error(f"Failed to fetch video info: {str(e)}")
            raise
        finally:
            self._progress_callback = None

    def _progress_hook(self, d):
        if self._progress_callback:
            status = ""
            if d["status"] == "downloading":
                status = f"Fetching video info... ({d.get('_percent_str', '0%')})"
            elif d["status"] == "finished":
                status = "Processing video info..."
            if status:
                self._progress_callback(status)

    def extract_frames(self, stream_url: str) -> Path:
        output_dir = Path(self.config["data_dir"])
        output_dir.mkdir(exist_ok=True)

        try:
            self.logger.info(f"Extracting frames from stream...")
            cap = cv2.VideoCapture(stream_url)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 25
                self.logger.warning("Could not determine FPS, defaulting to 25.")

            frame_interval = int(fps * self.config["frame_interval"])
            frame_count = 0
            saved_frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frame_filename = output_dir / f"frame{saved_frame_count:04d}.png"
                    cv2.imwrite(str(frame_filename), frame)
                    saved_frame_count += 1
                frame_count += 1

            cap.release()
            self.logger.info(f"Successfully extracted {saved_frame_count} frames.")
            return output_dir
        except Exception as e:
            self.logger.error(f"Failed to extract frames from stream: {str(e)}")
            raise

    def extract_captions(self) -> Path:
        """
        Extracts captions robustly using yt-dlp by forcing subtitle download.
        This bypasses the need for any other transcript library.
        """
        self.logger.info("Extracting captions using robust yt-dlp method...")
        caption_file_path = Path(self.config["data_dir"]) / f"captions_{self.video_id}.txt"
        
        # Define the output template for the caption file
        output_template = str(Path(self.config["data_dir"]) / f"{self.video_id}.%(ext)s")

        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US'],
            'subtitlesformat': 'vtt', # Specify VTT format
            'skip_download': True,
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.logger.info("Running yt-dlp to download subtitles...")
                ydl.extract_info(self.url, download=False)
                self.logger.info("yt-dlp finished execution.")

            # Search for the downloaded .vtt file
            vtt_file_path_str = str(Path(self.config["data_dir"]) / f"{self.video_id}.en.vtt")
            vtt_file = Path(vtt_file_path_str)

            if not vtt_file.exists():
                # Try finding any .vtt file as a fallback
                vtt_file = next(Path(self.config["data_dir"]).glob(f"{self.video_id}*.vtt"), None)

            if not vtt_file or not vtt_file.exists():
                self.logger.error(f"FATAL: yt-dlp did not create a caption file for video {self.video_id}.")
                caption_file_path.write_text("", encoding='utf-8')
                return caption_file_path

            self.logger.info(f"Found caption file: {vtt_file}")
            # Parse the VTT file
            with open(vtt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            caption_text = []
            for i, line in enumerate(lines):
                if "-->" in line:
                    try:
                        time_match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
                        if time_match:
                            start_time_str, end_time_str = time_match.groups()
                            
                            def to_seconds(t_str):
                                parts = t_str.split(':')
                                h, m = int(parts[0]), int(parts[1])
                                s, ms = map(int, parts[2].split('.'))
                                return h * 3600 + m * 60 + s + ms / 1000

                            start_sec = to_seconds(start_time_str)
                            end_sec = to_seconds(end_time_str)
                            
                            text = lines[i + 1].strip()
                            if text:
                                caption_text.append(f"<s> {start_sec:.2f} | {end_sec:.2f} | {text} </s>")
                    except Exception as parse_error:
                        self.logger.warning(f"Could not parse VTT line: {line.strip()} due to {parse_error}")
                        continue

            caption_file_path.write_text("\n".join(caption_text), encoding='utf-8')
            self.logger.info(f"Successfully parsed and saved {len(caption_text)} caption entries.")
            
            # Clean up the temporary .vtt file
            vtt_file.unlink()

        except Exception as e:
            self.logger.error(f"A critical error occurred during caption extraction with yt-dlp: {str(e)}")
            caption_file_path.write_text("", encoding='utf-8')

        return caption_file_path