import os
from google.cloud import storage
from googleapiclient.http import MediaFileUpload

gcs_client = storage.Client()
DRIVE_ID = os.environ.get("DRIVE_ID")

def find_existing_folder(drive_service, folder_name, parent_folder_id):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_folder_id}' in parents and trashed=false"
    results = drive_service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    items = results.get('files', [])
    if items:
        return items[0]['id']
    return None

def create_folder(drive_service, folder_name, parent_folder_id):
    print(f"Creating folder: {folder_name}")
    try:
        existing_folder = find_existing_folder(drive_service, folder_name, parent_folder_id)
        if existing_folder:
            return existing_folder
    except Exception as e:
        print(f"Error finding existing folder: {e}")
        pass

    folder_metadata = {
        'name': folder_name,
        'parents': [parent_folder_id],
        'mimeType': 'application/vnd.google-apps.folder',
        'driveId': DRIVE_ID
    }

    folder = drive_service.files().create(
        body=folder_metadata,
        fields='id',
        supportsAllDrives=True
    ).execute()

    return folder.get('id')

def get_existing_file_id(drive_service, file_name, folder_id):
    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(
        q=query,
        spaces='drive',
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    existing_files = results.get('files', [])
    return existing_files[0]['id'] if existing_files else None

def upload_new_file(drive_service, file_path, file_name, folder_id, drive_id=None):
    file_metadata = {
        'name': file_name,
        'parents': [folder_id],
        'driveId': drive_id
    }

    media = MediaFileUpload(file_path, resumable=True)
    new_file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True
    ).execute()

    print(f"Uploaded new file: {file_name} (ID: {new_file['id']})")
    return new_file['id']

def update_existing_file(drive_service, file_id, file_path):
    media = MediaFileUpload(file_path, resumable=True)
    updated_file = drive_service.files().update(
        fileId=file_id,
        media_body=media,
        supportsAllDrives=True
    ).execute()

    print(f"Updated existing file (ID: {updated_file['id']})")
    return updated_file['id']

def upload_file(drive_service, file_path, folder_id, drive_id=None):
    file_name = os.path.basename(file_path)
    existing_file_id = get_existing_file_id(drive_service, file_name, folder_id)

    if existing_file_id:
        return update_existing_file(drive_service, existing_file_id, file_path)
    return upload_new_file(drive_service, file_path, file_name, folder_id, drive_id)

def create_folder_rec(drive_service, folders, parent_folder_id):
    if len(folders) == 0:
        return parent_folder_id

    folder_name = folders.pop(0)
    folder_id = create_folder(drive_service, folder_name, parent_folder_id)
    return create_folder_rec(drive_service, folders, folder_id)

def delete_folder(drive_service, folder_id):
    drive_service.files().delete(
        fileId=folder_id,
        supportsAllDrives=True
    ).execute()

def upload_files(drive_service, local_folder_path, folder_id):
    for item in os.listdir(local_folder_path):
        print(item)
        item_path = os.path.join(local_folder_path, item)
        if not os.path.isdir(item_path):
            upload_file(drive_service, item_path, folder_id)

def list_gcs_files(bucket_name, folder):
    bucket = gcs_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    return blobs

def sync_gcs(bucket_name, gcs_folder, dest):
    blobs = list_gcs_files(bucket_name, gcs_folder)
    for blob in blobs:
        file_path = f'{dest}/{blob.name}'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            continue
        blob.download_to_filename(file_path)
