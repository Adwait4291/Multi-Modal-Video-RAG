import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import yt_dlp

# Correct import for youtube-transcript-api v1.2.2
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback imports
    try:
        import youtube_transcript_api
        YouTubeTranscriptApi = youtube_transcript_api.YouTubeTranscriptApi
        from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
    except ImportError:
        YouTubeTranscriptApi = None
        NoTranscriptFound = Exception
        TranscriptsDisabled = Exception


@dataclass
class VideoMetadata:
    title: str
    author: str
    views: int
    duration: int  # Video duration in seconds
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
        Extracts captions using the correct youtube-transcript-api method.
        Your version uses 'fetch' instead of 'get_transcript'.
        """
        caption_file = Path(self.config["data_dir"]) / f"captions_{self.video_id}.txt"
        caption_file.parent.mkdir(parents=True, exist_ok=True)

        if YouTubeTranscriptApi is None:
            self.logger.error("YouTubeTranscriptApi not available - check installation")
            caption_file.write_text("", encoding='utf-8')
            return caption_file

        try:
            self.logger.info(f"Attempting to extract captions for video: {self.video_id}")
            
            # Your version uses 'fetch' method instead of 'get_transcript'
            transcript = YouTubeTranscriptApi.fetch(
                self.video_id, 
                languages=['en', 'en-US', 'en-GB']
            )

            caption_text = []
            for entry in transcript:
                start = entry["start"]
                duration = entry.get("duration", 0)
                end = start + duration
                text = entry["text"].strip().replace('\n', ' ')
                caption_text.append(f"<s> {start:.2f} | {end:.2f} | {text} </s>")

            caption_file.write_text("\n".join(caption_text), encoding='utf-8')
            self.logger.info(f"Captions saved with {len(caption_text)} entries to {caption_file}")

        except NoTranscriptFound:
            self.logger.warning(f"No transcript found for video {self.video_id}")
            caption_file.write_text("", encoding='utf-8')

        except TranscriptsDisabled:
            self.logger.warning(f"Transcripts disabled for video {self.video_id}")
            caption_file.write_text("", encoding='utf-8')

        except Exception as e:
            self.logger.error(f"Unexpected error during caption extraction: {str(e)}")
            # Try fallback method using yt-dlp
            try:
                self.logger.info("Trying fallback method with yt-dlp...")
                caption_file = self._extract_captions_fallback()
            except Exception as fallback_error:
                self.logger.error(f"Fallback method also failed: {fallback_error}")
                caption_file.write_text("", encoding='utf-8')

        return caption_file

    def _extract_captions_fallback(self) -> Path:
        """
        Fallback method using yt-dlp to extract captions.
        """
        caption_file = Path(self.config["data_dir"]) / f"captions_{self.video_id}.txt"
        
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.url, download=False)
            
        # Check for available subtitles
        subtitles = info.get('subtitles', {})
        auto_subtitles = info.get('automatic_captions', {})
        
        if 'en' in subtitles or 'en' in auto_subtitles:
            caption_file.write_text("Captions available via yt-dlp (manual extraction needed)", encoding='utf-8')
            self.logger.info("Captions detected but manual extraction implementation needed")
        else:
            caption_file.write_text("", encoding='utf-8')
            self.logger.warning("No captions available")
            
        return caption_file

    def get_available_transcripts(self) -> List[Dict]:
        """
        Gets list of available transcripts for the video.
        Your version uses 'list' instead of 'list_transcripts'.
        """
        if YouTubeTranscriptApi is None:
            self.logger.error("YouTubeTranscriptApi not available")
            return []
            
        try:
            # Your version uses 'list' method instead of 'list_transcripts'
            transcript_list = YouTubeTranscriptApi.list(self.video_id)
            transcripts = [
                {
                    'language': t.language,
                    'language_code': t.language_code,
                    'is_generated': t.is_generated,
                    'is_translatable': t.is_translatable,
                }
                for t in transcript_list
            ]
            return transcripts
        except Exception as e:
            self.logger.error(f"Failed to get available transcripts: {e}")
            return []