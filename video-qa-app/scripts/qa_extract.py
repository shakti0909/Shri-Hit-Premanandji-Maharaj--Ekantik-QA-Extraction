import yt_dlp
from bs4 import BeautifulSoup
import pandas as pd
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
warnings.filterwarnings("ignore")

# Lock for thread-safe incrementing
_counter_lock = threading.Lock()


def get_playlist_videos(playlist_url, retries=2, delay=5):
    """
    Fetches all video IDs from a YouTube playlist efficiently.
    
    Args:
        playlist_url (str): YouTube playlist URL
        retries (int): Number of retries on failure (default: 2)
        delay (int): Delay in seconds between retries (default: 5 seconds)
        
    Returns:
        list: List of video IDs from the playlist, empty list if extraction fails
    """
    video_ids = []
    
    for attempt in range(retries):
        try:
            ydl_opts = {
                'extract_flat': True, 
                'quiet': False,
                'socket_timeout': 30,
                'skip_unavailable_fragments': True,
                # Subtitle options (though not used for playlist extraction)
                'write_subs': False,  # Don't download subs for playlist extraction
                'skip_download': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
                if 'entries' in info:
                    for entry in info['entries']:
                        if entry is not None and entry.get('id'):  # Skip None entries
                            video_ids.append(entry['id'])
                break  # Success, exit retry loop
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt + 1}/{retries} failed: {type(e).__name__}")
            if attempt < retries - 1:
                print(f"  ⏳ Waiting {delay}s before retry...")
                time.sleep(delay)
            else:
                print(f"  ❌ Skipping this playlist after {retries} retries")
    
    return video_ids


def process_single_playlist(args):
    """
    Processes a single playlist row (for parallel execution).
    
    Args:
        args (tuple): (idx, row, total_playlists) where row is a pandas Series
        
    Returns:
        tuple: (playlist_data_list, failed_playlist_info)
    """
    idx, row, total_playlists = args
    playlist_title = row['title']
    playlist_id = row['id']
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
    
    video_links = []
    failed_info = None
    
    print(f"📺 [{idx + 1}/{total_playlists}] Processing: {playlist_title}...")
    
    try:
        video_ids = get_playlist_videos(playlist_url, retries=2, delay=3)
        
        if not video_ids:
            print(f"  ⚠️  No videos extracted")
            failed_info = (playlist_title, "No videos extracted")
        else:
            print(f"  ✅ Found {len(video_ids)} videos")
            for video_id in video_ids:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                video_links.append({
                    'playlist_title': playlist_title,
                    'playlist_id': playlist_id,
                    'video_id': video_id,
                    'video_url': video_url
                })
    except Exception as e:
        print(f"  ❌ ERROR: {type(e).__name__}")
        failed_info = (playlist_title, str(type(e).__name__))
    
    return video_links, failed_info


def extract_videos_parallel(playlists_df, max_workers=3):
    """
    Extracts videos from multiple playlists in parallel.
    
    Args:
        playlists_df (pd.DataFrame): DataFrame with playlist data
        max_workers (int): Number of parallel workers (default: 3)
        
    Returns:
        tuple: (all_video_links, failed_playlists)
    """
    all_video_links = []
    failed_playlists = []
    
    # Prepare arguments for each playlist
    tasks = [(idx, row, len(playlists_df)) for idx, row in playlists_df.iterrows()]
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting parallel extraction with {max_workers} workers")
    print(f"{'='*60}\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_playlist, task): task for task in tasks}
        
        for future in as_completed(futures):
            video_links, failed_info = future.result()
            all_video_links.extend(video_links)
            if failed_info:
                failed_playlists.append(failed_info)
    
    return all_video_links, failed_playlists



def extract_qa_with_beautifulsoup(xml_transcript, chapters, video_id, video_url=""):
    """
    Parses XML transcripts using BeautifulSoup and filters into Q&A dictionaries.
    Applies exact 20-second timestamp buffer around each chapter.
    Generates timestamped URLs for direct navigation.
    
    Args:
        xml_transcript: XML caption content
        chapters (list): Chapter data with start_time, end_time, title
        video_id (str): YouTube video ID
        video_url (str): YouTube video URL (optional)
    
    Returns:
        list: Q&A dictionaries with video_id, video_url, timestamp_url, 
              question_number, start_time, end_time, question, answer_hindi, answer_english
    """
    soup = BeautifulSoup(xml_transcript, "lxml-xml") 
    captions = soup.find_all("text")
    
    final_transcript_qa = []
    
    for chapter_index, item in enumerate(chapters, start=1):
        start_time = int(item["start_time"])
        end_time = int(item["end_time"])
        
        # Generate timestamped URL for direct navigation to this question
        # Format: https://www.youtube.com/watch?v=VIDEO_ID&t=START_TIME
        timestamp_url = ""
        if video_url:
            timestamp_url = f"{video_url}&t={start_time}" if "?" in video_url else f"{video_url}?t={start_time}"
        
        # Build Q&A dictionary with both video_id and video_url
        qa_dict = {
            "video_id": video_id,
            "video_url": video_url,
            "timestamp_url": timestamp_url,
            "question_number": chapter_index,
            "start_time": start_time,
            "end_time": end_time,
            "question": item["title"],
            "answer_hindi": "",
            "answer_english": ""
        }
        
        for tag in captions:
            if 'start' not in tag.attrs:
                continue
                
            # Parse timestamp: extract integer seconds
            start_sv = int(tag['start'].split(".")[0])
            
            # Apply time buffer: 20s before to 20s after the chapter marker
            if start_sv >= (start_time - 20):
                qa_dict["answer_hindi"] += tag.text + " "
                
            if start_sv >= (start_time + 20):
                break
                
        final_transcript_qa.append(qa_dict)
        
    return final_transcript_qa

  

def save_qa_to_csv(qa_data_list, output_filename="youtube_qa_dataset.csv"):
    """Saves the extracted list of dictionaries to a CSV file."""
    df = pd.DataFrame(qa_data_list)
    df.to_csv(output_filename, index=False)
    print(f"✅ Extracted data for {len(df)} chapters successfully saved to {output_filename}")


def extract_video_metadata(video_url):
    """
    Extracts video metadata: chapters, captions (Hindi & English).
    
    Uses yt-dlp with subtitle extraction options:
    - --write-subs --skip-download: Download subtitles without video
    - --write-auto-subs: Fallback to autogenerated subtitles
    - --sub-langs hi,en: Extract Hindi and English subtitles
    - --sub-format srv1: XML format for BeautifulSoup parsing
    
    Falls back to automatic_captions API if subtitle files unavailable.
    
    Args:
        video_url (str): YouTube video URL
        
    Returns:
        dict: Contains 'chapters', 'captions_hi', 'captions_en', 'title'
    """
    try:
        ydl_opts = {
            'quiet': False,
            'no_warnings': False,
            'socket_timeout': 30,
            # Subtitle extraction options
            'write_subs': True,           # --write-subs
            'skip_download': True,        # --skip-download
            'write_auto_subs': True,      # --write-auto-subs (fallback on autogenerated)
            'sub_langs': ['hi', 'en'],    # --sub-lang hi,en
            'sub_format': 'srv1',         # --sub-format srv1 (XML format for parsing)
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            video_title = info.get('title', 'Unknown')
            chapters = info.get('chapters', [])
            
            # Extract captions (Hindi and English) - now using downloaded subtitle files
            captions_hi = []
            captions_en = []
            
            # yt-dlp downloads subtitle files, we need to read them
            import os
            import glob
            
            # Look for downloaded subtitle files
            video_id = video_url.split('v=')[-1] if 'v=' in video_url else video_url.split('/')[-1]
            
            # Find Hindi subtitle files (should be named like VIDEO_ID.hi.srv1)
            hi_files = glob.glob(f"*{video_id}*.hi.srv1") + glob.glob(f"*.hi.srv1")
            if hi_files:
                try:
                    with open(hi_files[0], 'r', encoding='utf-8') as f:
                        hi_content = f.read()
                    captions_hi_soup = BeautifulSoup(hi_content, "lxml-xml")
                    captions_hi_tags = captions_hi_soup.find_all("text")
                    if captions_hi_tags:
                        captions_hi = captions_hi_tags
                        print(f"  ✅ Hindi captions: {len(captions_hi)} entries")
                    else:
                        print(f"  ⚠️  Hindi captions file parsed but no text tags found")
                except Exception as e:
                    print(f"  ⚠️  Failed to read Hindi captions file: {type(e).__name__}")
            else:
                print(f"  ⚠️  No Hindi subtitle file found")
            
            # Find English subtitle files
            en_files = glob.glob(f"*{video_id}*.en.srv1") + glob.glob(f"*.en.srv1")
            if en_files:
                try:
                    with open(en_files[0], 'r', encoding='utf-8') as f:
                        en_content = f.read()
                    captions_en_soup = BeautifulSoup(en_content, "lxml-xml")
                    captions_en_tags = captions_en_soup.find_all("text")
                    if captions_en_tags:
                        captions_en = captions_en_tags
                        print(f"  ✅ English captions: {len(captions_en)} entries")
                    else:
                        print(f"  ⚠️  English captions file parsed but no text tags found")
                except Exception as e:
                    print(f"  ⚠️  Failed to read English captions file: {type(e).__name__}")
            else:
                print(f"  ⚠️  No English subtitle file found")
            
            # Fallback to automatic captions if subtitle files not found
            if not captions_hi or not captions_en:
                print(f"  🔄 Falling back to automatic captions...")
                auto_captions = info.get("automatic_captions", {})
                
                # Get Hindi captions from automatic_captions
                if not captions_hi and 'hi' in auto_captions:
                    for item in auto_captions['hi']:
                        if item['ext'] == 'srv1':
                            try:
                                vtt_response = requests.get(item['url'], timeout=10)
                                vtt_response.raise_for_status()
                                captions_hi_soup = BeautifulSoup(vtt_response.content, "lxml-xml")
                                captions_hi_tags = captions_hi_soup.find_all("text")
                                if captions_hi_tags:
                                    captions_hi = captions_hi_tags
                                    print(f"  ✅ Hindi captions (fallback): {len(captions_hi)} entries")
                                else:
                                    print(f"  ⚠️  Hindi captions parsed but no text tags found")
                            except Exception as e:
                                print(f"  ⚠️  Failed to extract Hindi captions: {type(e).__name__}")
                            break
                
                # Get English captions from automatic_captions
                if not captions_en and 'en' in auto_captions:
                    for item in auto_captions['en']:
                        if item['ext'] == 'srv1':
                            try:
                                vtt_response = requests.get(item['url'], timeout=10)
                                vtt_response.raise_for_status()
                                captions_en_soup = BeautifulSoup(vtt_response.content, "lxml-xml")
                                captions_en_tags = captions_en_soup.find_all("text")
                                if captions_en_tags:
                                    captions_en = captions_en_tags
                                    print(f"  ✅ English captions (fallback): {len(captions_en)} entries")
                                else:
                                    print(f"  ⚠️  English captions parsed but no text tags found")
                            except Exception as e:
                                print(f"  ⚠️  Failed to extract English captions: {type(e).__name__}")
                            break
            
            # Clean up downloaded subtitle files
            try:
                for file in hi_files + en_files:
                    if os.path.exists(file):
                        os.remove(file)
                        print(f"  🧹 Cleaned up: {file}")
            except Exception as e:
                print(f"  ⚠️  Failed to clean up subtitle files: {type(e).__name__}")
            
            return {
                'chapters': chapters,
                'captions_hi': captions_hi,
                'captions_en': captions_en,
                'title': video_title
            }
    except Exception as e:
        print(f"  ❌ Error extracting metadata: {type(e).__name__}")
        return None


def extract_qa_pairs_bilingual(video_url, video_title, chapters, captions_hi, captions_en):
    """
    Extracts Q&A pairs from chapters and captions (Hindi & English).
    Uses BeautifulSoup to parse XML caption timestamps and apply 20-second buffer.
    Generates timestamped URLs for direct navigation to each question.
    
    Args:
        video_url (str): Video URL
        video_title (str): Video title
        chapters (list): List of chapter dicts with start_time, end_time, title
        captions_hi (list): BeautifulSoup text tags for Hindi captions
        captions_en (list): BeautifulSoup text tags for English captions
        
    Returns:
        list: List of Q&A dictionaries with fields:
              video_id, video_url, timestamp_url, question_number, start_time, end_time, 
              question, answer_hindi, answer_english
    """
    # Extract video_id from URL
    video_id = video_url.split('v=')[-1] if 'v=' in video_url else video_url.split('/')[-1]
    
    qa_list = []
    
    if not chapters:
        return qa_list
    
    for idx, chapter in enumerate(chapters, start=1):
        start_time = int(chapter.get('start_time', 0))
        end_time = int(chapter.get('end_time', 0))
        question = chapter.get('title', f'Chapter {idx}')
        
        # Generate timestamped URL for direct navigation to this question
        # Format: https://www.youtube.com/watch?v=VIDEO_ID&t=START_TIME
        timestamp_url = f"{video_url}&t={start_time}" if "?" in video_url else f"{video_url}?t={start_time}"
        
        # Extract answer in Hindi (20s before to 20s after start_time) - using Prototype 1 logic
        answer_hi = ""
        if captions_hi:  # Only process if captions exist
            for tag in captions_hi:
                try:
                    if 'start' not in tag.attrs:
                        continue
                    # Parse timestamp: extract integer seconds from tag start (same as Prototype 1)
                    tag_start = int(tag['start'].split(".")[0])
                    # Capture captions from start-20 onward
                    if tag_start >= (start_time - 20):
                        answer_hi += tag.text + " "
                    # Stop when we reach start_time + 20
                    if tag_start >= (start_time + 20):
                        break
                except (ValueError, AttributeError, KeyError) as e:
                    continue  # Skip malformed tags
        
        # Extract answer in English (20s before to 20s after start_time) - using Prototype 1 logic
        answer_en = ""
        if captions_en:  # Only process if captions exist
            for tag in captions_en:
                try:
                    if 'start' not in tag.attrs:
                        continue
                    # Parse timestamp: extract integer seconds from tag start (same as Prototype 1)
                    tag_start = int(tag['start'].split(".")[0])
                    # Capture captions from start-20 onward
                    if tag_start >= (start_time - 20):
                        answer_en += tag.text + " "
                    # Stop when we reach start_time + 20
                    if tag_start >= (start_time + 20):
                        break
                except (ValueError, AttributeError, KeyError) as e:
                    continue  # Skip malformed tags
        
        qa_dict = {
            'video_id': video_id,
            'video_url': video_url,
            'timestamp_url': timestamp_url,
            'question_number': idx,
            'start_time': start_time,
            'end_time': end_time,
            'question': question,
            'answer_hindi': answer_hi.strip(),
            'answer_english': answer_en.strip()
        }
        
        qa_list.append(qa_dict)
    
    return qa_list


def process_single_video(args):
    """
    Processes a single video and extracts Q&A pairs (for parallel execution).
    
    Args:
        args (tuple): (idx, row, total_videos, unique_videos_df) where row is a pandas Series
        
    Returns:
        tuple: (qa_pairs_list, video_id, status)
    """
    idx, row, total_videos = args
    video_url = row['video_url']
    video_id = row['video_id']
    
    try:
        print(f"[{idx + 1}/{total_videos}] 📺 Processing: {video_id}", flush=True)
        
        # Extract video metadata
        metadata = extract_video_metadata(video_url)
        
        if not metadata:
            return [], video_id, "No metadata"
        
        chapters = metadata['chapters']
        captions_hi = metadata['captions_hi']
        captions_en = metadata['captions_en']
        video_title = metadata['title']
        
        if not chapters:
            return [], video_id, "No chapters"
        
        # Extract Q&A pairs
        qa_pairs = extract_qa_pairs_bilingual(video_url, video_title, chapters, captions_hi, captions_en)
        
        if qa_pairs:
            return qa_pairs, video_id, f"✅ {len(qa_pairs)} Q&A pairs"
        else:
            return [], video_id, "No Q&A extracted"
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)[:50]}"
        return [], video_id, error_msg


def extract_qa_parallel(videos_df, max_workers=4, show_progress=True):
    """
    Extracts Q&A pairs from multiple videos in parallel.
    
    Args:
        videos_df (pd.DataFrame): DataFrame with video URLs and IDs
        max_workers (int): Number of parallel workers (default: 4, increase for slower networks)
        show_progress (bool): Show progress messages
        
    Returns:
        tuple: (all_qa_pairs, failed_videos)
    """
    all_qa_pairs = []
    failed_videos = []
    successful_count = 0
    
    # Prepare tasks
    tasks = [(idx, row, len(videos_df)) for idx, row in videos_df.iterrows()]
    
    if show_progress:
        print(f"\n{'='*70}")
        print(f"🚀 Starting parallel extraction with {max_workers} workers")
        print(f"📊 Total videos to process: {len(videos_df)}")
        print(f"{'='*70}\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, task): task for task in tasks}
        
        for future in as_completed(futures):
            qa_pairs, video_id, status = future.result()
            
            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)
                successful_count += 1
                if show_progress:
                    print(f"  {status}", flush=True)
            else:
                failed_videos.append((video_id, status))
                if show_progress:
                    print(f"  ⚠️  {video_id}: {status}", flush=True)
    
    if show_progress:
        print(f"\n{'='*70}")
        print(f"✅ Parallel extraction complete!")
        print(f"{'='*70}")
    
    return all_qa_pairs, failed_videos