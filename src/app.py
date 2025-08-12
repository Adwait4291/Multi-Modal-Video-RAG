import logging
import os
import time
from pathlib import Path

import streamlit as st
import yaml
from inference import InferenceProcessor
from retriever import VideoRetriever
from utils.helper import cleanup_data_directories
from utils.logger import setup_logger
from video_indexer import VideoIndexer
from video_processor import VideoProcessor


# Setup logger
logger = setup_logger()

# Custom CSS styling
STYLE = """
<style>
    /* Main app background */
    .main {
        background-color: #F0F2F6; /* Light grey background */
    }

    /* Title styling */
    h1 {
        color: #1E3A8A; /* Deep blue for titles */
        border-bottom: 3px solid #1E3A8A;
        padding-bottom: 10px;
    }

    /* Header styling */
    h2, h3 {
        color: #374151; /* Dark grey for headers */
    }

    /* Text input box: Fixed invisible text issue */
    .stTextInput>div>div>input {
        border: 2px solid #D1D5DB; /* Grey border */
        background-color: #FFFFFF; /* White background */
        color: #111827; /* Dark text color for visibility */
        border-radius: 8px;
    }

    /* Button styling */
    .stButton>button {
        background-color: #2563EB; /* Vibrant blue */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1D4ED8; /* Darker blue on hover */
    }

    /* Progress bar styling */
    .stProgress>div>div>div {
        background-color: #2563EB; /* Vibrant blue */
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
    }

    /* Log box styling */
    .log-box {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        background-color: #FFFFFF;
        border-left: 5px solid #2563EB; /* Blue accent line */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* API Key Popup */
    .api-key-popup {
        background-color: white; 
        padding: 30px; 
        border-radius: 10px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
"""


def load_config():
    """
    Load and parse the application configuration from the YAML file.
    """
    config_path = "config/config.yaml"
    logger.info(f"Loading configuration from {config_path}")

    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
            logger.debug(f"Successfully loaded configuration: {config}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
        raise

def init_session_state():
    """
    Initialize the Streamlit session state with required keys.
    """
    logger.debug("Initializing session state variables")
    
    required_keys = {
        "video_url": None,
        "index": None,
        "retriever": None,
        "video_id": None,
        "inference_processor": None,
        "gemini_key": None,
    }

    for key, default_value in required_keys.items():
        if key not in st.session_state:
            logger.debug(f"Initializing session state key: {key}")
            st.session_state[key] = default_value


def main():

    st_time = time.time()

    logger.info("Starting Video RAG System application")
    st.set_page_config(page_title="Video RAG System", layout="wide", page_icon="🎥")
    st.markdown(STYLE, unsafe_allow_html=True)

    init_session_state()

    # API Key Popup
    if not st.session_state.gemini_key:
        logger.info("No Gemini API key found - displaying key input form")
        with st.container():
            st.markdown("<div class='api-key-popup'>", unsafe_allow_html=True)
            st.header("🔑 Gemini API Key Required")
            api_key = st.text_input(
                "Please enter your Gemini API key to continue:", type="password"
            )
            if st.button("Submit Key"):
                if api_key:
                    logger.info("API key submitted successfully")
                    st.session_state.gemini_key = api_key
                    st.rerun()
                else:
                    logger.warning("Empty API key submitted")
                    st.error("Please enter a valid API key")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

    # Load configuration and initialize processor
    config = load_config()
    if st.session_state.inference_processor is None:
        try:
            logger.info("Initializing InferenceProcessor")
            st.session_state.inference_processor = InferenceProcessor(st.session_state.gemini_key)
        except Exception as e:
            logger.error(f"Failed to initialize InferenceProcessor: {str(e)}")
            st.error(f"❌ Failed to initialize API: {str(e)}")
            del st.session_state.gemini_key
            st.rerun()

    # Sidebar
    st.sidebar.header("Settings ⚙️")
    
    # --- ADDED: Sliders for top_k configuration ---
    st.sidebar.subheader("Retrieval Settings")
    similarity_top_k = st.sidebar.slider(
        "Relevant Text Chunks (Top K)",
        min_value=1,
        max_value=10,
        value=5,
        help="Controls how many of the most similar text segments are retrieved for a query."
    )
    image_similarity_top_k = st.sidebar.slider(
        "Relevant Image Frames (Top K)",
        min_value=1,
        max_value=10,
        value=5,
        help="Controls how many of the most similar image frames are retrieved for a query."
    )
    # --- END ADDED ---

    if st.sidebar.button("🧹 Cleanup All Data"):
        try:
            logger.info("Starting cleanup of data directories")
            with st.spinner("Cleaning up previous data..."):
                cleanup_data_directories()
            for key in ["video_url", "index", "retriever", "video_id"]:
                st.session_state[key] = None
            logger.info("Cleanup completed successfully")
            st.success("All previous data cleaned successfully!")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
            st.error(f"Error during cleanup: {str(e)}")

    # Main Content
    st.title("🎥 Video RAG System")

    # Video Processing Section
    st.header("Step 1: Process YouTube Video 🎬")
    video_url = st.text_input("Enter YouTube URL:")
    
    if st.button("🚀 Process Video") and video_url:
        logger.info(f"Starting video processing for URL: {video_url}")
        try:
            status_container = st.container()
            progress_bar = status_container.progress(0)
            log_box = status_container.empty()

            def update_log(message, progress):
                logger.debug(f"Processing status: {message}")
                log_box.markdown(f'<div class="log-box">📌 {message}</div>', unsafe_allow_html=True)
                progress_bar.progress(progress)

            with st.spinner("Processing video..."):
                update_log("Initializing video processor...", 5)
                video_processor = VideoProcessor(video_url, config)

                update_log("Fetching video info...", 10)
                metadata, stream_url = video_processor.get_video_info()
                
                def format_duration(seconds):
                    return f"{seconds//60}:{seconds%60:02d}"
                video_duration = format_duration(metadata.duration)
                update_log(f"Video info fetched - Duration: {video_duration}", 25)

                update_log("Extracting frames from video stream...", 30)
                frames_dir = video_processor.extract_frames(stream_url)
                update_log(f"Extracted {len(list(frames_dir.glob('*.png')))} frames", 50)

                update_log("Extracting video captions...", 60)
                captions_path = video_processor.extract_captions()
                update_log(f"Captions saved to: {captions_path}", 70)

                update_log("Creating multimodal index...", 80)
                indexer = VideoIndexer(config)
                index = indexer.create_multimodal_index(frames_dir, captions_path, video_processor.video_id)
                update_log("Index creation complete", 90)

                st.session_state.index = index
                st.session_state.video_url = video_url
                st.session_state.video_id = video_processor.video_id
                
                # --- UPDATED: Pass slider values to retriever ---
                st.session_state.retriever = VideoRetriever(
                    index, 
                    similarity_top_k=similarity_top_k, 
                    image_similarity_top_k=image_similarity_top_k
                )
                # --- END UPDATED ---

                update_log("Video processed successfully!", 100)
                logger.info("Video processing completed successfully")
                st.success("✅ Video processed and ready to be queried!")

        except Exception as e:
            error_msg = f"❌ Error processing video: {str(e)}"
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            st.error(error_msg)

    # Query Section
    if st.session_state.index:
        st.header("Step 2: Query Video Content 🔍")
        if st.session_state.video_url:
            st.video(st.session_state.video_url)

        query = st.text_input("Enter your query about the video:")
        if st.button("📤 Submit Query"):
            if not query:
                st.warning("Please enter a query.")
            else:
                logger.info(f"Processing query: {query}")
                try:
                    with st.spinner("Analyzing your query..."):
                        retrieved_images, retrieved_texts = st.session_state.retriever.retrieve(query)
                        st.info(f"Found {len(retrieved_images)} relevant frames and {len(retrieved_texts)} text segments.")
                        
                        response = st.session_state.inference_processor.process_query(
                            retrieved_images, retrieved_texts, query
                        )

                    st.subheader("💡 Answer")
                    st.markdown(response['answer'])

                    if retrieved_images:
                        st.subheader("🖼️ Retrieved Frames")
                        num_cols = min(config.get("max_display_frames", 3), len(retrieved_images))
                        cols = st.columns(num_cols)
                        for idx, image_path in enumerate(retrieved_images):
                            with cols[idx % num_cols]:
                                st.image(str(image_path), use_container_width=True, caption=f"Frame {idx + 1}")

                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}", exc_info=True)
                    st.error(f"❌ Error processing query: {str(e)}")

    et = time.time()
    logger.info(f"Page rendered in {round(et - st_time, 3)} seconds")


if __name__ == "__main__":
    main()