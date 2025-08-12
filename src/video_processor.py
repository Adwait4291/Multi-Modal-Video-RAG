import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

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
        # This is the crucial step: create an instance of the API class
        self.transcript_api = YouTubeTranscriptApi()

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
    caption_file = Path(self.config["data_dir"]) / f"captions_{self.video_id}.txt"

    try:
        self.logger.info(f"Attempting to fetch transcripts for video: {self.video_id}")

        # Detect API method version
        if hasattr(self.transcript_api, "list_transcripts"):
            # Modern API flow
            transcript_list = self.transcript_api.list_transcripts(self.video_id)
            try:
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
            except NoTranscriptFound:
                try:
                    transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                except NoTranscriptFound:
                    transcript = next(iter(transcript_list), None)

            if transcript:
                srt_data = transcript.fetch()

        elif hasattr(self.transcript_api, "list"):
            # Legacy API flow
            transcript_list = self.transcript_api.list(self.video_id)
            transcript = None
            for t in transcript_list:
                if t.get("language_code") in ['en', 'en-US', 'en-GB']:
                    transcript = t
                    break
            if not transcript and transcript_list:
                transcript = transcript_list[0]

            if transcript:
                srt_data = self.transcript_api.fetch(self.video_id, transcript['language_code'])

        else:
            self.logger.error("youtube-transcript-api has no transcript listing method.")
            caption_file.write_text("", encoding='utf-8')
            return caption_file

        if not transcript:
            self.logger.warning(f"No transcript found for video {self.video_id}")
            caption_file.write_text("", encoding='utf-8')
            return caption_file

        # Save captions in <s> format
        caption_text = []
        for entry in srt_data:
            start = entry.get("start", 0)
            duration = entry.get("duration", 0)
            end = start + duration
            text = entry.get("text", "").strip().replace('\n', ' ')
            caption_text.append(f"<s> {start:.2f} | {end:.2f} | {text} </s>")

        caption_file.write_text("\n".join(caption_text), encoding='utf-8')
        self.logger.info(f"Captions saved with {len(caption_text)} entries to {caption_file}")

    except TranscriptsDisabled:
        self.logger.error(f"Transcripts are disabled for video {self.video_id}")
        caption_file.write_text("", encoding='utf-8')
    except Exception as e:
        self.logger.error(f"Unexpected error fetching captions: {e}")
        caption_file.write_text("", encoding='utf-8')

    return caption_file


def get_available_transcripts(self) -> List[Dict]:
    try:
        if hasattr(self.transcript_api, "list_transcripts"):
            transcript_list_obj = self.transcript_api.list_transcripts(self.video_id)
            return [
                {
                    'language': t.language,
                    'language_code': t.language_code,
                    'is_generated': t.is_generated,
                    'is_translatable': t.is_translatable,
                }
                for t in transcript_list_obj
            ]
        elif hasattr(self.transcript_api, "list"):
            transcript_list_obj = self.transcript_api.list(self.video_id)
            return [
                {
                    'language': t.get("language"),
                    'language_code': t.get("language_code"),
                    'is_generated': t.get("is_generated", False),
                    'is_translatable': t.get("is_translatable", False),
                }
                for t in transcript_list_obj
            ]
        else:
            self.logger.error("youtube-transcript-api has no transcript listing method.")
            return []
    except Exception as e:
        self.logger.error(f"Failed to get available transcripts: {e}")
        return []
