import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable

import cv2
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from moviepy.editor import VideoFileClip

print("Loaded video_processor.py from:", __file__)


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
        self._progress_callback: Optional[Callable] = None

    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL with improved error handling."""
        try:
            # Handle different YouTube URL formats
            if "?v=" in url:
                return url.split("?v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]
            elif "youtube.com/embed/" in url:
                return url.split("/embed/")[1].split("?")[0]
            elif len(url) == 11 and url.isalnum():  # Direct video ID
                return url
            else:
                raise ValueError("Invalid YouTube URL format")
        except Exception as e:
            self.logger.error(f"Failed to extract video ID from URL: {url} - {e}")
            raise ValueError(f"Invalid YouTube URL: {e}")

    def get_video_info(self, progress_callback: Optional[Callable] = None) -> Tuple[VideoMetadata, str]:
        """Get video metadata and stream URL with improved error handling."""
        self._progress_callback = progress_callback
        ydl_opts = {
            "format": "best[height<=720]",  # Limit quality to avoid issues
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [self._progress_hook],
            "extract_flat": False,
            "ignoreerrors": False,
        }

        try:
            self.logger.info(f"Fetching video info for: {self.url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)

            if not info:
                raise ValueError("Could not extract video information")

            # Calculate filesize if available
            filesize_mb = 0
            if info.get("filesize"):
                filesize_mb = info["filesize"] / (1024 * 1024)
            elif info.get("filesize_approx"):
                filesize_mb = info["filesize_approx"] / (1024 * 1024)

            metadata = VideoMetadata(
                title=info.get("title", "Unknown"),
                author=info.get("uploader", "Unknown"),
                views=info.get("view_count", 0) or 0,
                duration=int(info.get("duration", 0) or 0),
                filesize_mb=filesize_mb,
                format=info.get("format", "Unknown"),
                resolution=f"{info.get('width', 'Unknown')}x{info.get('height', 'Unknown')}",
                video_id=self.video_id,
            )

            stream_url = info.get('url')
            if not stream_url:
                raise ValueError("Could not extract stream URL from video info")

            self.logger.info("Successfully fetched video info and stream URL.")
            return metadata, stream_url

        except yt_dlp.DownloadError as e:
            self.logger.error(f"yt-dlp download error: {str(e)}")
            raise ValueError(f"Failed to fetch video info: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to fetch video info: {str(e)}")
            raise
        finally:
            self._progress_callback = None

    def _progress_hook(self, d):
        """Progress hook for yt-dlp with safe callback handling."""
        if self._progress_callback and callable(self._progress_callback):
            try:
                status = ""
                if d.get("status") == "downloading":
                    percent = d.get("_percent_str", "0%")
                    status = f"Fetching video info... ({percent})"
                elif d.get("status") == "finished":
                    status = "Processing video info..."

                if status:
                    self._progress_callback(status)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")

    def extract_frames(self, video_path: Path) -> Path:
        """Extract frames from local video file with improved error handling."""
        output_dir = Path(self.config["data_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        clip = None
        try:
            self.logger.info(f"Extracting frames from video file...")

            clip = VideoFileClip(str(video_path))
            fps = clip.fps
            if fps <= 0:
                fps = 25
                self.logger.warning("Could not determine FPS, defaulting to 25.")

            frame_interval = max(1, int(fps * self.config.get("frame_interval", 1)))
            saved_frame_count = 0

            for i, frame in enumerate(clip.iter_frames()):
                if i % frame_interval == 0:
                    frame_filename = output_dir / f"frame{saved_frame_count:04d}.png"
                    success = cv2.imwrite(str(frame_filename), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if success:
                        saved_frame_count += 1
                    else:
                        self.logger.warning(f"Failed to save frame {saved_frame_count}")

            self.logger.info(f"Successfully extracted {saved_frame_count} frames.")
            return output_dir

        except Exception as e:
            self.logger.error(f"Failed to extract frames from video file: {str(e)}")
            raise ValueError(f"Frame extraction failed: {str(e)}")
        finally:
            if clip is not None:
                clip.close()
    
    def extract_captions(self) -> Path:
        """
        Extract captions/transcript from YouTube video.
        Compatible with youtube-transcript-api==1.2.2
        """
        caption_file = Path(self.config["data_dir"]) / f"captions_{self.video_id}.txt"
        
        try:
            self.logger.info(f"Attempting to fetch transcripts for video: {self.video_id}")
            
            # For version 1.2.2, this is the correct way - create instance and use fetch()
            yt_api = YouTubeTranscriptApi()
            fetched_transcript = yt_api.fetch(self.video_id, languages=['en', 'en-US', 'en-GB'])
            
            # Convert to raw data (list of dictionaries)
            srt_data = fetched_transcript.to_raw_data()
            
            self.logger.info(f"Retrieved transcript for video {self.video_id}")

            # Process and save transcript data
            caption_text = []
            for i, entry in enumerate(srt_data):
                try:
                    start = float(entry.get("start", 0))
                    duration = float(entry.get("duration", 0))
                    end = start + duration
                    text = entry.get("text", "").strip().replace('\n', ' ').replace('\r', ' ')
                    
                    if text:  # Only add non-empty text
                        caption_text.append(f"<s> {start:.2f} | {end:.2f} | {text} </s>")
                except Exception as entry_error:
                    self.logger.warning(f"Error processing transcript entry {i}: {entry_error}")
                    continue

            if caption_text:
                caption_file.write_text("\n".join(caption_text), encoding='utf-8')
                self.logger.info(f"Captions saved with {len(caption_text)} entries to {caption_file}")
            else:
                self.logger.warning("No valid caption entries found")
                caption_file.write_text("", encoding='utf-8')

        except NoTranscriptFound:
            self.logger.warning(f"No transcripts found for video {self.video_id} in specified languages.")
            caption_file.write_text("", encoding='utf-8')
        except TranscriptsDisabled:
            self.logger.error(f"Transcripts are disabled for video {self.video_id}")
            caption_file.write_text("", encoding='utf-8')
        except Exception as e:
            self.logger.error(f"Unexpected error fetching captions: {e}")
            caption_file.write_text("", encoding='utf-8')
        
        return caption_file

    def get_available_transcripts(self) -> List[Dict]:
        """Get list of available transcripts for the video."""
        try:
            # Try multiple approaches to get available transcripts
            transcript_list = None
            
            # Approach 1: Static method
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
            except AttributeError:
                pass
            
            # Approach 2: Instance method
            if transcript_list is None:
                try:
                    api = YouTubeTranscriptApi()
                    transcript_list = api.list_transcripts(self.video_id)
                except AttributeError:
                    pass
            
            if transcript_list is None:
                self.logger.warning("Could not access transcript listing functionality")
                return []
            
            available = []
            for transcript in transcript_list:
                available.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                })
            
            self.logger.info(f"Found {len(available)} available transcripts")
            return available
            
        except Exception as e:
            self.logger.error(f"Error getting available transcripts: {e}")
            return []