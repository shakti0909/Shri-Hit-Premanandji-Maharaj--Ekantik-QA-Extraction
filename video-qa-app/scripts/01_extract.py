import yt_dlp
from bs4 import BeautifulSoup
import pandas as pd


def get_playlist_videos(playlist_url):
    """Fetches all video IDs from a YouTube playlist efficiently."""
    ydl_opts = {
        'extract_flat': True, 
        'quiet': True
    }
    video_ids = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        if 'entries' in info:
            for entry in info['entries']:
                video_ids.append(entry['id'])
    return video_ids



def extract_qa_with_beautifulsoup(xml_transcript, chapters, video_id):
    """Parses XML transcripts using BeautifulSoup and filters into a Q&A dictionary."""
    soup = BeautifulSoup(xml_transcript, "lxml-xml") 
    captions = soup.find_all("text")
    
    final_transcript_qa = []
    
    for chapter_index, item in enumerate(chapters, start=1):
        start_time = int(item["start_time"])
        end_time = int(item["end_time"])
        
        # Build the filtered dictionary and inject the video_id
        qa_dict = {
            "video_id": video_id,
            "question_number": chapter_index,
            "start_time": start_time,
            "end_time": end_time,
            "question": item["title"],
            "answer_text": ""
        }
        
        for tag in captions:
            if 'start' not in tag.attrs:
                continue
                
            # Split logic to safely get integer seconds
            start_sv = int(tag['start'].split(".")[0])
            
            # Apply time buffer: 20s before to 20s after the chapter marker
            if start_sv >= (start_time - 20):
                qa_dict["answer_text"] += tag.text + " "
                
            if start_sv >= (start_time + 20):
                break
                
        final_transcript_qa.append(qa_dict)
        
    return final_transcript_qa

  

def save_qa_to_csv(qa_data_list, output_filename="youtube_qa_dataset.csv"):
    """Saves the extracted list of dictionaries to a CSV file."""
    df = pd.DataFrame(qa_data_list)
    df.to_csv(output_filename, index=False)
    print(f"✅ Extracted data for {len(df)} chapters successfully saved to {output_filename}")