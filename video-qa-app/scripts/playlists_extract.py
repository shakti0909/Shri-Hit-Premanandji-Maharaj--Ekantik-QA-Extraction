import yt_dlp

def get_playlists_by_channel_and_keyword(channel_name, keyword):
    """
    Extracts playlist titles and IDs from a YouTube channel that contain a specific keyword.
    
    Args:
        channel_name (str): The channel handle (e.g., '@ChannelName')
        keyword (str): Keyword to filter playlist titles
        
    Returns:
        list: List of dictionaries with 'title' and 'id' for matching playlists
    """
    channel_url = f"https://www.youtube.com/{channel_name}/playlists"
    
    ydl_opts = {
        'extract_flat': 'in_playlist',  # Extract playlist info without videos
        'quiet': True,
        'no_warnings': True,
    }
    
    playlists = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            if 'entries' in info:
                for entry in info['entries']:
                    title = entry.get('title', '')
                    if keyword.lower() in title.lower():
                        playlists.append({
                            'title': title,
                            'id': entry.get('id', '')
                        })
    except Exception as e:
        print(f"Error extracting playlists: {e}")
    
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