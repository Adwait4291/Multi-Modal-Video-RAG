import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


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
        """
        Fetches video metadata and a direct stream URL without downloading the file.
        """
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
                filesize_mb=0,  # Not applicable for streaming
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
        """
        Extracts frames from a video stream using OpenCV.
        """
        output_dir = Path(self.config["data_dir"])
        output_dir.mkdir(exist_ok=True)

        try:
            self.logger.info(f"Extracting frames from stream...")
            cap = cv2.VideoCapture(stream_url)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:  # Handle potential issues with getting FPS
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
        Extract captions/transcript from YouTube video with multiple fallback options.
        """
        try:
            self.logger.info(f"Extracting captions for video: {self.video_id}")
            
            srt_data = None
            try:
                # Get available transcripts
                transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
                
                # Try to find English transcript first
                try:
                    transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                    self.logger.info("Found English transcript.")
                except:
                    # If no English, try auto-generated
                    try:
                        transcript = transcript_list.find_generated_transcript(['en'])
                        self.logger.info("Found auto-generated English transcript.")
                    except:
                        # Get any available transcript
                        transcript = next(iter(transcript_list))
                        self.logger.info(f"Using first available transcript: {transcript.language}")
                
                srt_data = transcript.fetch()
            
            except Exception as e:
                self.logger.warning(f"Failed to get transcript via list method: {e}. Falling back to direct method.")
                # Fallback: Try direct method
                try:
                    srt_data = YouTubeTranscriptApi.get_transcript(self.video_id, languages=['en', 'en-US'])
                except Exception as e2:
                    self.logger.warning(f"Failed to get transcript via direct method: {e2}. Trying without language preference.")
                    # Try without language specification as a last resort
                    srt_data = YouTubeTranscriptApi.get_transcript(self.video_id)

            # Create caption file
            caption_file = Path(self.config["data_dir"]) / f"captions_{self.video_id}.txt"
            caption_text = []

            # Process transcript data
            for entry in srt_data:
                start = entry["start"]
                duration = entry.get("duration", 0)
                end = start + duration
                text = entry["text"].strip().replace('\n', ' ')
                caption_text.append(f"<s> {start:.2f} | {end:.2f} | {text} </s>")

            # Save to file
            caption_file.write_text("\n".join(caption_text), encoding='utf-8')
            self.logger.info(f"Captions saved with {len(caption_text)} entries to {caption_file}")
            return caption_file
            
        except Exception as e:
            self.logger.error(f"Could not extract any captions: {str(e)}")
            # Create empty caption file as a final fallback
            caption_file = Path(self.config["data_dir"]) / f"captions_{self.video_id}.txt"
            caption_file.write_text("", encoding='utf-8')
            self.logger.warning(f"Created an empty caption file as a fallback: {caption_file}")
            return caption_file

    def get_available_transcripts(self) -> List[Dict]:
        """
        Get list of available transcripts for debugging purposes.
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
            transcripts = []
            
            for transcript in transcript_list:
                transcripts.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                })
            
            return transcripts
        except Exception as e:
            self.logger.error(f"Failed to get available transcripts: {e}")
            return []