from youtube_transcript_api import YouTubeTranscriptApi

def test_api_methods():
    """Test the correct API methods for your version"""
    
    # Test video ID (Rick Roll - usually has captions)
    video_id = "dQw4w9WgXcQ"
    
    print("Testing YouTubeTranscriptApi.fetch()...")
    try:
        # Try the fetch method with just video_id
        transcript = YouTubeTranscriptApi.fetch(video_id)
        print(f"✓ Success with fetch(video_id): Got {len(transcript)} entries")
        if transcript:
            print(f"  Sample entry: {transcript[0]}")
    except Exception as e:
        print(f"✗ fetch(video_id) failed: {e}")
        
        # Try with languages parameter
        try:
            transcript = YouTubeTranscriptApi.fetch(video_id, languages=['en'])
            print(f"✓ Success with fetch(video_id, languages): Got {len(transcript)} entries")
        except Exception as e2:
            print(f"✗ fetch(video_id, languages) also failed: {e2}")
    
    print("\nTesting YouTubeTranscriptApi.list()...")
    try:
        transcript_list = YouTubeTranscriptApi.list(video_id)
        print(f"✓ Success with list(): Got transcript list")
        print(f"  Available transcripts: {len(list(transcript_list))}")
    except Exception as e:
        print(f"✗ list() failed: {e}")
    
    # Check method signatures
    print(f"\nMethod signatures:")
    import inspect
    
    if hasattr(YouTubeTranscriptApi, 'fetch'):
        fetch_sig = inspect.signature(YouTubeTranscriptApi.fetch)
        print(f"  fetch{fetch_sig}")
    
    if hasattr(YouTubeTranscriptApi, 'list'):
        list_sig = inspect.signature(YouTubeTranscriptApi.list)
        print(f"  list{list_sig}")

if __name__ == "__main__":
    test_api_methods()