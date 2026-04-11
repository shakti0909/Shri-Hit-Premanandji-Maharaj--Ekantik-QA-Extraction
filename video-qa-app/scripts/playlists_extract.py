import yt_dlp
import time

def get_playlists_by_channel_and_keyword(channel_name, keyword, retries=3, delay=2):
    """
    Extracts playlist titles and IDs from a YouTube channel that contain a specific keyword.
    
    Args:
        channel_name (str): The channel handle (e.g., '@ChannelName')
        keyword (str): Keyword to filter playlist titles
        retries (int): Number of retries on failure (default: 3)
        delay (int): Delay in seconds between retries (default: 2 seconds)
        
    Returns:
        list: List of dictionaries with 'title' and 'id' for matching playlists
    """
    channel_url = f"https://www.youtube.com/{channel_name}/playlists"
    
    ydl_opts = {
        'extract_flat': 'in_playlist',  # Extract playlist info without videos
        'quiet': False,
        'no_warnings': False,
        'socket_timeout': 30,
    }
    
    playlists = []
    for attempt in range(retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                if 'entries' in info:
                    for entry in info['entries']:
                        if entry is not None:  # Skip None entries
                            title = entry.get('title', '')
                            playlist_id = entry.get('id', '')
                            if title and playlist_id and keyword.lower() in title.lower():
                                playlists.append({
                                    'title': title,
                                    'id': playlist_id
                                })
                    break  # Success, exit retry loop
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} - Error extracting playlists: {e}")
            if attempt < retries - 1:
                print(f"⏳ Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print("❌ Failed to extract playlists after all retries")
    
    return playlists

def save_playlists_to_csv(playlists, output_filename="playlists.csv"):
    """
    Saves the extracted playlists to a CSV file.
    
    Args:
        playlists (list): List of playlist dictionaries
        output_filename (str): Output CSV filename
    """
    import pandas as pd
    df = pd.DataFrame(playlists)
    df.to_csv(output_filename, index=False)
    print(f"✅ Saved {len(df)} playlists to {output_filename}")