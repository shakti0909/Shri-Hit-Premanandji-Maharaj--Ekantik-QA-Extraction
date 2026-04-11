import yt_dlp
from bs4 import BeautifulSoup
import pandas as pd
import time
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
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


def parse_srt(srt_file_path):
    """
    Parses an SRT file into a list of caption dictionaries.
    
    Args:
        srt_file_path (str): Path to the SRT file
        
    Returns:
        list: List of dicts with 'start', 'end', 'text'
    """
    captions = []
    try:
        with open(srt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # SRT format: blocks separated by double newlines
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.split('\n')
            if len(lines) < 3:
                continue
            
            # Parse timestamp line: "00:00:01,000 --> 00:00:04,000"
            timestamp_line = lines[1]
            if '-->' not in timestamp_line:
                continue
            
            start_str, end_str = timestamp_line.split(' --> ')
            
            # Convert to seconds (float)
            def time_to_seconds(time_str):
                h, m, s = time_str.replace(',', '.').split(':')
                return int(h) * 3600 + int(m) * 60 + float(s)
            
            start = time_to_seconds(start_str)
            end = time_to_seconds(end_str)
            
            # Combine text lines
            text = ' '.join(lines[2:])
            
            captions.append({
                'start': str(int(start)),  # Convert to string for compatibility
                'end': str(int(end)),
                'text': text
            })
    except Exception as e:
        print(f"  ⚠️  Failed to parse SRT: {type(e).__name__}")
    
    return captions


# def extract_metadata_hybrid(video_url):
#     """
#     Extracts chapters via yt-dlp and captions via youtube-transcript-api.
#     Bypasses ffmpeg and JS runtime requirements.
#     """
#     video_id = video_url.split('v=')[-1].split('&')[0] if 'v=' in video_url else video_url.split('/')[-1]

#     ydl_opts = {
#         'quiet': True,
#         'skip_download': True,
#         'writesubtitles': False,
#         'extract_flat': False
#     }

#     chapters = []
#     video_title = 'Unknown'

#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info = ydl.extract_info(video_url, download=False)
#             chapters = info.get('chapters', [])
#             video_title = info.get('title', 'Unknown')
#     except Exception as e:
#         print(f"  ⚠️ Metadata error for {video_id}: {e}")

#     captions_hi = []
#     captions_en = []

#     try:
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

#         try:
#             captions_hi = transcript_list.find_transcript(['hi']).fetch()
#             print(f"  ✅ Hindi captions: {len(captions_hi)} blocks")
#         except Exception:
#             print("  ⚠️ No Hindi captions found via API")

#         try:
#             captions_en = transcript_list.find_transcript(['en']).fetch()
#             print(f"  ✅ English captions: {len(captions_en)} blocks")
#         except Exception:
#             print("  ⚠️ No English captions found via API")

#     except Exception as e:
#         print(f"  ⚠️ Transcript API error: {e}")

#     return {
#         'chapters': chapters,
#         'captions_hi': captions_hi,
#         'captions_en': captions_en,
#         'title': video_title,
#         'video_id': video_id
#     }

def extract_metadata_hybrid(video_url):
    """
    Extracts chapters via yt-dlp and captions via youtube-transcript-api.
    Uses get_transcript for maximum compatibility.
    """
    video_id = video_url.split('v=')[-1].split('&')[0] if 'v=' in video_url else video_url.split('/')[-1]

    # 1. yt-dlp Metadata Extraction
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'writesubtitles': False,
        'extract_flat': False
    }

    chapters = []
    video_title = 'Unknown'

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            chapters = info.get('chapters', [])
            video_title = info.get('title', 'Unknown')
    except Exception as e:
        print(f"  ⚠️ Metadata error for {video_id}: {e}")

    # 2. Official Transcript API Extraction (Backward Compatible)
    captions_hi = []
    captions_en = []

    try:
        # Fetch Hindi explicitly
        try:
            captions_hi = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
            print(f"  ✅ Hindi captions: {len(captions_hi)} blocks")
        except Exception:
            print("  ⚠️ No Hindi captions found via API")

        # Fetch English explicitly
        try:
            captions_en = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            print(f"  ✅ English captions: {len(captions_en)} blocks")
        except Exception:
            print("  ⚠️ No English captions found via API")

    except Exception as e:
        print(f"  ⚠️ Transcript API error: {e}")

    return {
        'chapters': chapters,
        'captions_hi': captions_hi,
        'captions_en': captions_en,
        'title': video_title,
        'video_id': video_id
    }

def extract_qa_pairs_hybrid(video_url, metadata):
    """
    Maps the youtube-transcript-api format to Q&A chapter extraction logic.
    """
    if not metadata or not metadata.get('chapters'):
        return []

    video_id = metadata['video_id']
    chapters = metadata['chapters']
    captions_hi = metadata.get('captions_hi', [])
    captions_en = metadata.get('captions_en', [])

    qa_list = []

    for idx, chapter in enumerate(chapters, start=1):
        start_time = int(chapter.get('start_time', 0))
        end_time = int(chapter.get('end_time', 0))
        question = chapter.get('title', f'Chapter {idx}')
        timestamp_url = f"{video_url}&t={start_time}" if "?" in video_url else f"{video_url}?t={start_time}"

        buffer_start = start_time - 20
        buffer_end = start_time + 20

        answer_hi = ""
        for cap in captions_hi:
            try:
                if cap['start'] >= buffer_start and cap['start'] <= buffer_end:
                    answer_hi += cap['text'] + ' '
                elif cap['start'] > buffer_end:
                    break
            except KeyError:
                continue

        answer_en = ""
        for cap in captions_en:
            try:
                if cap['start'] >= buffer_start and cap['start'] <= buffer_end:
                    answer_en += cap['text'] + ' '
                elif cap['start'] > buffer_end:
                    break
            except KeyError:
                continue

        qa_list.append({
            'video_id': video_id,
            'video_url': video_url,
            'timestamp_url': timestamp_url,
            'question_number': idx,
            'start_time': start_time,
            'end_time': end_time,
            'question': question,
            'answer_hindi': answer_hi.strip(),
            'answer_english': answer_en.strip()
        })

    return qa_list


# Keep the old interface names working by aliasing them to the hybrid flow
extract_video_metadata = extract_metadata_hybrid
extract_qa_pairs_bilingual = extract_qa_pairs_hybrid


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


import yt_dlp
import pandas as pd

def extract_chapters_only(video_url):
    """
    Extracts only chapter titles (Questions) and timestamps using yt-dlp.
    Bypasses all caption, ffmpeg, and JS runtime dependencies.
    """
    video_id = video_url.split('v=')[-1].split('&')[0] if 'v=' in video_url else video_url.split('/')[-1]
    
    # Minimal options: Do not download video, do not download subs
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'writesubtitles': False,
        'extract_flat': False # Required to get chapter metadata for individual videos
    }
    
    qa_list = []
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract only the metadata JSON
            info = ydl.extract_info(video_url, download=False)
            chapters = info.get('chapters')
            
            if not chapters:
                print(f"  ⚠️ No chapters found for {video_id}")
                return []
            
            print(f"  ✅ Found {len(chapters)} chapters for {video_id}")
            
            for idx, chapter in enumerate(chapters, start=1):
                start_time = int(chapter.get('start_time', 0))
                end_time = int(chapter.get('end_time', 0))
                question = chapter.get('title', f'Chapter {idx}')
                
                # Construct the direct jump link
                timestamp_url = f"{video_url}&t={start_time}" if "?" in video_url else f"{video_url}?t={start_time}"
                
                qa_list.append({
                    'video_id': video_id,
                    'video_url': video_url,
                    'timestamp_url': timestamp_url,
                    'question_number': idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'question': question
                })
                
    except Exception as e:
        print(f"  ❌ Error processing {video_id}: {e}")
        
    return qa_list

# --- Execution Example ---
# video_urls = ["https://www.youtube.com/watch?v=y6es3rtMolo", "https://www.youtube.com/watch?v=-aDTTy-o7vE"]
# all_data = []
# for url in video_urls:
#     all_data.extend(extract_chapters_only(url))
# 
# df = pd.DataFrame(all_data)
# df.to_csv("dataset_v1_chapters_only.csv", index=False)
# display(df.head())