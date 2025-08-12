import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable

import cv2
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

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

    def extract_frames(self, stream_url: str, progress_callback: Optional[Callable] = None) -> Path:
        """Extract frames from video stream with improved error handling."""
        output_dir = Path(self.config["data_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = None
        try:
            self.logger.info(f"Extracting frames from stream...")

            # Try to open the video stream
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise ValueError(f"Could not open video stream: {stream_url}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                fps = 25
                self.logger.warning("Could not determine FPS, defaulting to 25.")

            frame_interval = max(1, int(fps * self.config.get("frame_interval", 1)))
            frame_count = 0
            saved_frame_count = 0

            self.logger.info(f"Video FPS: {fps}, Total frames: {total_frames}, Frame interval: {frame_interval}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_filename = output_dir / f"frame{saved_frame_count:04d}.png"
                    success = cv2.imwrite(str(frame_filename), frame)
                    if success:
                        saved_frame_count += 1
                    else:
                        self.logger.warning(f"Failed to save frame {saved_frame_count}")

                frame_count += 1

                # Progress callback
                if progress_callback and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(f"Extracting frames... {progress:.1f}%")

            self.logger.info(f"Successfully extracted {saved_frame_count} frames from {frame_count} total frames.")
            return output_dir

        except Exception as e:
            self.logger.error(f"Failed to extract frames from stream: {str(e)}")
            raise ValueError(f"Frame extraction failed: {str(e)}")
        finally:
            if cap is not None:
                cap.release()

    def extract_captions(self, target_language: Optional[str] = None) -> Path:
        """Extract captions with optional translation and comprehensive error handling."""
        output_dir = Path(self.config["data_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        if target_language:
            caption_file = output_dir / f"captions_{self.video_id}_{target_language}.txt"
        else:
            caption_file = output_dir / f"captions_{self.video_id}.txt"

        try:
            self.logger.info(f"Attempting to fetch transcripts for video: {self.video_id}")

            # Use corrected original caption fetching, compatible with v1.2.2
            if target_language:
                srt_data = self._fetch_and_translate_captions(self.video_id, target_language)
            else:
                srt_data = self._fetch_original_captions()

            # Process and save transcript data
            if srt_data:
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
            else:
                self.logger.warning(f"No transcript data available for video {self.video_id}")
                caption_file.write_text("", encoding='utf-8')

        except TranscriptsDisabled:
            self.logger.error(f"Transcripts are disabled for video {self.video_id}")
            caption_file.write_text("", encoding='utf-8')
        except NoTranscriptFound:
            self.logger.error(f"No transcript found for video {self.video_id}")
            caption_file.write_text("", encoding='utf-8')
        except Exception as e:
            self.logger.error(f"Unexpected error fetching captions: {e}")
            caption_file.write_text("", encoding='utf-8')

        return caption_file

    def _fetch_original_captions(self) -> List[Dict]:
        """Fetch original captions using the get_transcript method compatible with v1.2.2."""
        try:
            # Use get_transcript directly, specifying language preference.
            # This method will automatically fall back to auto-generated if manual is not found.
            self.logger.info(f"Using legacy get_transcript method for video: {self.video_id}")
            srt_data = YouTubeTranscriptApi.get_transcript(self.video_id, languages=['en', 'en-US', 'en-GB'])
            self.logger.info(f"Retrieved transcript using get_transcript for video {self.video_id}")
            return srt_data

        except TranscriptsDisabled:
            self.logger.error(f"Transcripts are disabled for video {self.video_id}")
            raise
        except NoTranscriptFound:
            self.logger.warning(f"No transcripts found for video {self.video_id} in specified languages.")
            return []
        except Exception as e:
            self.logger.error(f"Failed to fetch original captions: {e}")
            raise

    def _fetch_and_translate_captions(self, video_id: str, target_lang: str) -> List[Dict]:
        """Fetches a transcript and translates it to the target language."""
        try:
            # Step 1: List all available transcripts using a compatible method
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Step 2: Find a suitable source transcript (e.g., English)
            try:
                transcript = transcript_list.find_transcript(['en', 'en-US'])
                self.logger.info("Found English transcript for translation")
            except NoTranscriptFound:
                # If no English transcript is available, try to find any available transcript
                self.logger.info("No English transcript found, trying to find any available language.")
                transcript = next(iter(transcript_list), None)
                if not transcript:
                    self.logger.warning("No transcripts found in any language.")
                    return []
                self.logger.info(f"Using {transcript.language} transcript for translation")

            # Step 3: Translate the found transcript to the target language
            translated_transcript = transcript.translate(target_lang)

            # Step 4: Fetch and return the translated data
            translated_data = translated_transcript.fetch()
            self.logger.info(f"Successfully translated captions to {translated_transcript.language} ({translated_transcript.language_code}).")
            return translated_data

        except TranscriptsDisabled:
            self.logger.error(f"Transcripts are disabled for video {video_id}.")
            return []
        except Exception as e:
            self.logger.error(f"Translation error occurred: {e}")
            return []

    def get_available_transcripts(self) -> List[Dict]:
        """Get list of available transcripts with improved error handling."""
        try:
            # This method will not work with v1.2.2 as list_transcripts is not available.
            # You would need to upgrade the library for this functionality.
            if hasattr(YouTubeTranscriptApi, "list_transcripts"):
                transcript_list_obj = YouTubeTranscriptApi.list_transcripts(self.video_id)
                return [
                    {
                        'language': getattr(t, 'language', 'Unknown'),
                        'language_code': getattr(t, 'language_code', 'unknown'),
                        'is_generated': getattr(t, 'is_generated', False),
                        'is_translatable': getattr(t, 'is_translatable', False),
                    }
                    for t in transcript_list_obj
                ]
            else:
                self.logger.warning("Your youtube-transcript-api version does not support list_transcripts.")
                return []
        except TranscriptsDisabled:
            self.logger.warning(f"Transcripts are disabled for video {self.video_id}")
            return []
        except NoTranscriptFound:
            self.logger.warning(f"No transcripts found for video {self.video_id}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to get available transcripts: {e}")
            return []