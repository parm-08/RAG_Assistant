# RAG AI Teaching Assistant

An end-to-end Retrieval-Augmented Generation (RAG) based AI Teaching Assistant that processes educational video content and enables intelligent question answering over transcribed lectures.

This system converts videos → audio → transcripts → embeddings → semantic retrieval → LLM-generated responses.

 Project Overview

This assistant allows users to:

-Upload lecture videos

-Automatically transcribe audio

-Convert transcripts into embeddings

-Store vectors efficiently

-Retrieve relevant lecture content

-Generate context-aware answers using an LLM

-It is designed to act as a personal AI tutor over your own learning materials.

 Project Pipeline
Step 1 – Collect Videos
Move all your lecture/video files into the audios/ or designated input folder.

Step 2 – Convert Video to MP3
Run:
video_to_mp3.py
This converts all video files into MP3 format for transcription.

Step 3 – Convert MP3 to JSON (Transcription)
Run:
mp3_to_json.py

This:
Uses Whisper for speech-to-text
Converts audio into structured JSON transcripts
Stores transcripts inside the jsons/ folder

Step 4 – Convert JSON to Vector Embeddings
Run:
preprocess_json.py

This:
Reads transcript JSON files
Splits text into chunks
Generates embeddings
Saves results into embeddings.joblib
The joblib file contains:
Text chunks
Corresponding embeddings

Step 5 – Retrieval + Prompt + LLM Response
When a user asks a question:
Load embeddings.joblib
Convert user query into embedding
Perform similarity search
Retrieve top-k relevant chunks
Construct prompt using retrieved context

Project Structure
RAG_BASED_ASSISTANT/
│
├── audios/                 # Input audio files
├── jsons/                  # Transcribed JSON files
├── newjsons/               # Processed JSON files
├── whisper/                # Whisper model files
│
├── config.py               # Configuration settings
├── mp3_to_json.py          # Audio → JSON transcription
├── preprocess_json.py      # JSON → Embeddings
├── merge_chunks.py         # Chunk merging utility
├── processing_incoming.py  # Query processing
├── embeddings.joblib       # Stored embeddings
├── prompt.txt              # Prompt template
├── response.txt            # LLM output
└── readme.md


TECH STACK
Python
Whisper (Speech-to-Text)
FAISS / Vector similarity search
LLaMA 3.2 (LLM)
Joblib (Embedding storage)
NumPy / Pandas

Key Features

- Lecture video processing

- Automatic transcription

- Smart chunking strategy

- Semantic search retrieval

- Context-aware AI answers

- Efficient embedding storage

-Send prompt to LLM

-Generate final response

<img width="1040" height="206" alt="image" src="https://github.com/user-attachments/assets/4b3b2f04-bfeb-4158-a3e3-67cdb912dd14" />

<img width="1477" height="181" alt="image" src="https://github.com/user-attachments/assets/07096b68-3fae-4cf0-9898-4d74bd7bc4db" />


