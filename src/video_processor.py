import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import yt_dlp
from moviepy.video.io.VideoFileClip import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


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

    def download_video(self, progress_callback=None) -> Tuple[VideoMetadata, Path]:
        filename = f"{self.video_id}.mp4"
        output_path = Path(self.config["video_dir"]) / filename

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(output_path),
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [self._progress_hook],
        }

        self._progress_callback = progress_callback

        try:
            self.logger.info(f"Starting download of video: {self.url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)

            if not output_path.exists():
                raise FileNotFoundError(f"Download failed - no file created at {output_path}")

            with VideoFileClip(str(output_path)) as clip:
                duration = int(clip.duration)

            metadata = VideoMetadata(
                title=info.get("title", "Unknown"),
                author=info.get("uploader", "Unknown"),
                views=info.get("view_count", 0),
                duration=duration,
                filesize_mb=round(output_path.stat().st_size / (1024 * 1024), 2),
                format=info.get("format", "Unknown"),
                resolution=f"{info.get('width', 'Unknown')}x{info.get('height', 'Unknown')}",
                video_id=self.video_id,
            )
            return metadata, output_path
        except Exception as e:
            self.logger.error(f"Failed to download video: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            raise
        finally:
            self._progress_callback = None

    def _progress_hook(self, d):
        if self._progress_callback:
            status = ""
            if d["status"] == "downloading":
                status = f"Downloading video... ({d.get('_percent_str', '0%')})"
            elif d["status"] == "finished":
                status = "Processing downloaded video..."

            if status:
                self._progress_callback(status)

    def extract_frames(self, video_path: Path) -> Path:
        output_dir = Path(self.config["data_dir"])
        output_dir.mkdir(exist_ok=True)

        try:
            self.logger.info(f"Extracting frames from video: {video_path}")
            with VideoFileClip(str(video_path)) as clip:
                fps = 1 / self.config["frame_interval"]
                clip.write_images_sequence(str(output_dir / "frame%04d.png"), fps=fps, logger=None)
            return output_dir
        except Exception as e:
            self.logger.error(f"Failed to extract frames: {str(e)}")
            raise

    def extract_captions(self) -> Path:
        """
        Extract captions/transcript from YouTube video with multiple fallback options.
        """
        try:
            self.logger.info(f"Extracting captions for video: {self.video_id}")
            
            # Method 1: Try to get transcript with language preferences
            transcript_list = None
            transcript = None
            
            try:
                # Get available transcripts
                transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
                
                # Try to find English transcript first
                try:
                    transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                    self.logger.info("Found English transcript")
                except:
                    # If no English, try auto-generated
                    try:
                        transcript = transcript_list.find_generated_transcript(['en'])
                        self.logger.info("Found auto-generated English transcript")
                    except:
                        # Get any available transcript
                        transcript = next(iter(transcript_list))
                        self.logger.info(f"Using available transcript: {transcript.language}")
                
                # Fetch the actual transcript data
                srt_data = transcript.fetch()
                
            except Exception as e:
                self.logger.error(f"Failed to get transcript via list method: {e}")
                # Fallback: Try direct method (older API style)
                try:
                    srt_data = YouTubeTranscriptApi.get_transcript(self.video_id, languages=['en', 'en-US'])
                except Exception as e2:
                    self.logger.error(f"Failed to get transcript via direct method: {e2}")
                    # Try without language specification
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
            self.logger.error(f"Failed to extract captions: {str(e)}")
            # Create empty caption file as fallback
            caption_file = Path(self.config["data_dir"]) / f"captions_{self.video_id}.txt"
            caption_file.write_text("", encoding='utf-8')
            self.logger.warning(f"Created empty caption file: {caption_file}")
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