import sys
import os
import traceback

print("--- Debugging SoulX-Singer Imports ---")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Add soulx_singer_repo to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
soulx_repo_path = os.path.join(current_dir, "soulx_singer_repo")

print(f"Expected repo path: {soulx_repo_path}")

if os.path.exists(soulx_repo_path):
    print("✅ Repo path exists.")
    if soulx_repo_path not in sys.path:
        print(f"Adding to sys.path: {soulx_repo_path}")
        sys.path.insert(0, soulx_repo_path)
else:
    print("❌ Repo path DOES NOT exist! This is likely the problem.")

print("\n--- 1. Attempting to import 'preprocess' package ---")
try:
    import preprocess
    print(f"✅ 'preprocess' package imported from: {preprocess.__file__}")
except Exception:
    print("❌ Failed to import 'preprocess'")
    traceback.print_exc()

print("\n--- 2. Attempting to import 'NoteTranscriber' dependencies ---")
print("Importing librosa...")
try:
    import librosa
    print("✅ librosa ok")
except:
    print("❌ librosa failed")
    traceback.print_exc()

print("Importing matplotlib...")
try:
    import matplotlib
    import matplotlib.pyplot
    print("✅ matplotlib ok")
except:
    print("❌ matplotlib failed")
    traceback.print_exc()

print("Importing pretty_midi...")
try:
    import pretty_midi
    print("✅ pretty_midi ok")
except:
    print("❌ pretty_midi failed")
    traceback.print_exc()

print("\n--- 3. Attempting to import NoteTranscriber ---")
try:
    from preprocess.tools.note_transcription.model import NoteTranscriber
    print("✅ Success! NoteTranscriber imported.")
except Exception:
    print("❌ Failed to import NoteTranscriber")
    traceback.print_exc()

print("\n--- End Debug ---")
