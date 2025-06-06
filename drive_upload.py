from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
import os

# Define the scope to allow file creation and access
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Path to your downloaded service account key JSON file
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")  # <- Replace with your actual file path

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def upload_file_to_drive(file_path, drive_filename, folder_id=None):
    service = get_drive_service()
    file_metadata = {'name': drive_filename}
    if folder_id:
        file_metadata['parents'] = [folder_id]  # Puts file in the target folder
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')

    # Make public
    service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()

    # Print URL for debug / info
    print(f"Uploaded file to Drive: https://drive.google.com/file/d/{file_id}/view")

    return file_id

def make_file_public(file_id):
    service = get_drive_service()
    service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()
    return f"https://drive.google.com/uc?id={file_id}"

def delete_file_from_drive(file_id):
    service = get_drive_service()
    service.files().delete(fileId=file_id).execute()
    print(f"Deleted file from Drive: {file_id}")
