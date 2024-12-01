---
layout: post
title: Web Devel - Storing Data on Google Drive 
date: '2023-11-01 13:19'
subtitle: Google Drive
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - Web Devel
---

For personal projects, all AWS services will start charging once one service starts. Google drive on the other hand, gives everybody 5GB of storage for free. So for infrequent data read / write, Google drive could be a good option.

## API Setup

1. Go to [manage your Google Drive API](https://console.cloud.google.com/marketplace/product/google/drive.googleapis.com?project=probable-byway-439515-h8)

2. Go to APIs & Services > Credentials in the Google Cloud Console.

3. Create OAuth Client ID

    - Click on Create Credentials -> OAuth client ID.
    - If prompted to configure the consent screen, do so:
        - Choose External for user type.
        - Fill in the required details like App name, User support email, etc.
        - Under Scopes, you can add scopes if needed, but for basic access, the default is sufficient.
        - Save and continue through the remaining steps.
    - After configuring the consent screen, proceed to create the OAuth client ID.
    - Select Desktop app as the application type.
    - Name your client (e.g., "Drive API Client") and click Create.
    - After creation, a dialog will appear with your Client ID and Client Secret.
    - Click Download JSON. This file (e.g., credentials.json) contains your OAuth 2.0 credentials.

4. From the left-hand menu, go to APIs & Services > OAuth consent screen.
    - Scroll down to the Test users section.
    - Click on Add Users.
    - Enter the email address you want to add (e.g., your personal or work email).

## Programmatic Interface Setup

Install dependencies

```bash
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

Minimal working example

```python
import os
import pickle
import sys
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def upload_file(service, file_path, mime_type=None, parent_id=None):
    """Uploads a file to Google Drive."""
    file_name = os.path.basename(file_path)
    file_metadata = {'name': file_name}
    if parent_id:
        file_metadata['parents'] = [parent_id]
    media = MediaFileUpload(file_path, mimetype=mime_type if mime_type else 'application/octet-stream')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File ID: {file.get("id")}')

def authenticate():
    """Authenticates the user and returns the Drive service."""
    creds = None
    # The file token.pickle stores the user's access and refresh tokens.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("credentials.json not found. Please ensure you have downloaded it from Google Cloud Console.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_to_drive.py <file_path> [mime_type] [parent_folder_id]")
        sys.exit(1)

    file_path = sys.argv[1]
    mime_type = sys.argv[2] if len(sys.argv) > 2 else None
    parent_id = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        sys.exit(1)

    service = authenticate()
    upload_file(service, file_path, mime_type, parent_id)

if __name__ == '__main__':
    main()
```
