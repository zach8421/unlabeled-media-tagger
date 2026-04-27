# Setup for end users

This guide takes you from an empty Windows or Mac laptop to a working
pipeline that downloads media from Google Drive, finds and clusters faces,
and produces a review package you can share.

You do not need to know Python, git, or Google Cloud to follow this. You
do need:

- Admin rights on your computer (you will be installing software).
- A Google account that can read the Drive folder you want to process.
- About 60–90 minutes the first time. Most of that is waiting for
  installs and downloads.

Follow the steps in order. Each step has the exact command to run and what
you should see when it works.

---

## Step 1 — Install Git

Git is the tool that downloads the project's source code.

### Mac

Macs usually have Git already, bundled with Apple's developer tools. Open
**Terminal** (press `Cmd+Space`, type `Terminal`, press Return) and run:

```bash
git --version
```

- If you see something like `git version 2.39.5` you are done with this
  step.
- If you see a popup that says *"The git command requires the command
  line developer tools. Would you like to install the tools now?"*, click
  **Install** and wait for it to finish (a few minutes). Then run
  `git --version` again to confirm.

### Windows

Download the installer from <https://git-scm.com/downloads> and run it.
The defaults are fine — you can click **Next** through every screen.

After it finishes, open the Start menu, type `Git Bash`, and confirm a
new black terminal window opens. Close it for now; you will use a
different terminal later.

---

## Step 2 — Install Miniforge

Miniforge is a small Python installer designed for scientific computing.
It is the most reliable way to get this pipeline running on a fresh
machine.

Go to <https://conda-forge.org/download/> and download the installer for
your system:

- **Mac (Apple Silicon, M1/M2/M3/M4)**: pick the `arm64` Mac installer.
- **Mac (Intel)**: pick the `x86_64` Mac installer. If you do not know
  which Mac you have, click the Apple menu → *About This Mac*. Look for
  *Chip* (Apple Silicon) or *Processor* (Intel).
- **Windows**: pick the `x86_64` Windows installer (`.exe`).

### Mac install

The Mac download is a `.sh` script, not a clickable installer. In
Terminal, run the installer using a wildcard so the exact filename does
not matter:

```bash
bash ~/Downloads/Miniforge3-MacOSX-*.sh
```

If you have multiple Miniforge installers in your Downloads folder
(unlikely on a fresh machine), this command will fail — delete the old
ones first and try again.

Press Return to scroll the license, type `yes` to accept, press Return
again to accept the install location, and finally type `yes` when it
asks whether to **initialize Miniforge3**. That last `yes` is important
— it makes the `conda` command available in future Terminal sessions.

Close Terminal completely, then open a new Terminal window. You should
now see `(base)` at the start of your prompt. That is how you know
Miniforge is installed and active.

### Windows install

Double-click the downloaded `.exe`. Click through the installer:

- "Just Me" install (the default) is fine.
- The default install location is fine.
- On the *Advanced Options* page, **leave the boxes at their defaults**.
  In particular, do not check "Add Miniforge3 to my PATH environment
  variable" — the documentation says that is not recommended, and you
  will use a dedicated terminal (next step) instead.

When the installer finishes, open the Start menu and confirm you see
**Miniforge Prompt** as a new entry. You will use that in the next step.

---

## Step 3 — Open the right terminal

Every command in this guide runs inside a terminal where `conda` and
`python` work. The terminal you use depends on your operating system.

### Mac

Open **Terminal**. Confirm your prompt starts with `(base)`. If it does
not, you skipped the `yes` in step 2 — re-run the Miniforge installer
and answer `yes` when it asks about initialization.

### Windows

Open **Miniforge Prompt** from the Start menu. Do **not** use PowerShell
or Command Prompt for this guide — `conda` may not be available there.
Do not use Git Bash either; it does not handle conda environments
cleanly.

You should see a prompt like:

```text
(base) C:\Users\YourName>
```

The `(base)` at the start is what tells you conda is active.

---

## Step 4 — Download the project

In your terminal, change into a folder where you want the project to
live (your home folder is fine), then clone the repository.

**Mac:**

```bash
cd ~
git clone https://github.com/zach8421/unlabeled-media-tagger.git
cd unlabeled-media-tagger
```

**Windows:**

```bash
cd %USERPROFILE%
git clone https://github.com/zach8421/unlabeled-media-tagger.git
cd unlabeled-media-tagger
```

When `git clone` works, you will see a few lines about *Cloning into
'unlabeled-media-tagger'…*, then *Resolving deltas…*, then your prompt
returns. The project is now in a folder called `unlabeled-media-tagger`
under wherever you ran the command.

The `cd unlabeled-media-tagger` step changes into that folder. **Every
remaining command in this guide assumes you are inside this folder.** If
you close your terminal and come back later, run `cd path/to/unlabeled-media-tagger`
first.

---

## Step 5 — Create the Python environment

This step installs Python and all the libraries the pipeline needs into
an isolated environment. The exact commands are different on Mac and
Windows.

### Mac

Run:

```bash
conda env create -f environment.yml
```

This will take 5–15 minutes. You will see a wall of text as conda
downloads and installs packages. When it finishes, you should see
something like *"To activate this environment, use: conda activate
unlabeled-media-tagger"*.

If the install fails halfway through with package-resolution errors,
delete the partially created environment with `conda env remove -n
unlabeled-media-tagger` and try again. If it still fails, fall through
to the Windows instructions below — they also work on Mac as a fallback.

### Windows

The `environment.yml` file pins Mac-specific package versions and will
not work on Windows. Use this three-command sequence instead:

```bash
conda create -n unlabeled-media-tagger python=3.10
conda activate unlabeled-media-tagger
pip install -e .
pip install -r requirements.txt
```

The first command creates an empty environment with Python 3.10 — say
`y` when it asks to proceed. This takes about a minute.

`conda activate unlabeled-media-tagger` switches into that environment.
Your prompt should change from `(base)` to `(unlabeled-media-tagger)`.

`pip install -e .` installs the project itself plus its core
dependencies (Google Drive client, OpenCV, DeepFace). This takes 5–10
minutes — DeepFace pulls in TensorFlow, which is large.

`pip install -r requirements.txt` adds a few extras the project needs
(notably `tf-keras`, which is required for newer TensorFlow versions).
This is fast.

If you later see an error during the first pipeline run that mentions
**`tf-keras` is required** or **No module named `tf_keras`**, run
`pip install tf-keras` from inside the activated environment and try
again.

---

## Step 6 — Activate the environment

If you just finished step 5 on Mac, run:

```bash
conda activate unlabeled-media-tagger
```

Windows users already activated it during step 5.

Either way, your prompt should now start with
`(unlabeled-media-tagger)`. That is how you know you are in the right
environment.

**You will need to repeat this `conda activate` step every time you
open a new terminal window.** It is not permanent. If you ever see
errors like `python: command not found` or `No module named
unlabeled_media_tagger`, the most common cause is that you forgot to
activate the environment.

---

## Step 7 — Set up Google credentials

This step is the longest and the most fiddly. The pipeline needs
permission to read your Google Drive folder, and Google requires you to
set up your own credentials for that. There is no shortcut around the
Google Cloud Console — but you only need to do this once.

> Google's Cloud Console interface changes occasionally. The labels and
> page layouts described below may not match exactly, but the steps —
> create a project, enable the Drive API, configure consent, create a
> Desktop OAuth client, download the JSON — are durable. If a button has
> moved, look for the same words nearby.

### 7a. Open the Google Cloud Console

Go to <https://console.cloud.google.com/> and sign in with the Google
account that has access to the Drive folders you want to process.

The first time, Google may ask you to accept the terms of service for
Google Cloud. Accept them.

### 7b. Create a project

At the top of the page, next to the *Google Cloud* logo, there is a
project picker (it will say *Select a project* or show an existing
project name). Click it.

In the dialog that opens, click **NEW PROJECT** in the top right. Give
it a name like `Media Tagger` (the name does not matter, only you will
see it). You can leave the *Organization* and *Location* fields at
their defaults. Click **CREATE**.

Wait a few seconds for the project to be created, then make sure it is
selected in the project picker at the top.

### 7c. Enable the Google Drive API

In the search bar at the top of the Console, type `Google Drive API`.
Click the result that says **Google Drive API** under *Marketplace*.

You will land on a page with a blue **ENABLE** button. Click it. After
a few seconds the page changes to show the API is enabled.

### 7d. Configure the OAuth consent screen

In the left navigation (you may need to click the hamburger menu icon
in the top left to reveal it), go to **APIs & Services → OAuth consent
screen**.

You will be asked to choose a *User Type*. Pick **External** and click
**CREATE**.

Fill in the required fields on the *App information* page:

- **App name**: anything you like (`Media Tagger` is fine).
- **User support email**: your own email address.
- **Developer contact information → Email addresses**: your own email
  address again.

You can leave every other field blank. Click **SAVE AND CONTINUE**.

On the *Scopes* page, click **SAVE AND CONTINUE** without adding any
scopes. The pipeline will request the scope it needs at runtime.

On the *Test users* page, click **+ ADD USERS** and add the email
address of the Google account you will use to log in (the same one
that owns the Drive folder you want to process). Click **SAVE AND
CONTINUE**.

On the *Summary* page, click **BACK TO DASHBOARD**.

### 7e. Create the OAuth client ID

In the left navigation, go to **APIs & Services → Credentials**.

Click **+ CREATE CREDENTIALS** at the top, then choose **OAuth client
ID** from the dropdown.

For *Application type*, pick **Desktop app**. Give it a name like
`Media Tagger Desktop` (again, only you see this).

Click **CREATE**. A dialog appears with your client ID and client
secret. Click **DOWNLOAD JSON** to save the credentials file. Then
click **OK**.

### 7f. Save the credentials in the project

The downloaded file will have a long name like
`client_secret_1234567890-abcdefg.apps.googleusercontent.com.json`. The
pipeline expects it at a specific path with a specific name.

In your terminal, inside the `unlabeled-media-tagger` folder, create
the `secrets` folder:

```bash
mkdir secrets
```

(On Windows in Miniforge Prompt, the `mkdir` command works the same
way.)

Then move the downloaded JSON into that folder and rename it to
`credentials.json`. You can do this with your file manager (Finder on
Mac, File Explorer on Windows) — drag the file into the `secrets`
folder and rename it. Or in your terminal:

```bash
# Mac
mv ~/Downloads/client_secret_*.json secrets/credentials.json

# Windows
move %USERPROFILE%\Downloads\client_secret_*.json secrets\credentials.json
```

The end result must be a file at exactly:

```text
unlabeled-media-tagger/secrets/credentials.json
```

If the pipeline later complains that it cannot find
`secrets/credentials.json`, it almost always means the file was saved
in the wrong location or with the wrong name (often the original long
`client_secret_…json` name). The fix is to rename and move it.

The `secrets/` folder is excluded from version control — your
credentials never leave your computer.

---

## Step 8 — Your first pipeline run

You are ready to run the pipeline.

### Pick a small test folder

Find a Google Drive folder with **1–3 photos** in it. A small folder
keeps the first run fast and makes the output easy to inspect. You can
use a real folder for now; the pipeline does not modify the files
themselves unless you pass `--write-drive-descriptions`.

Get the folder's URL: in Google Drive, open the folder and copy the URL
from your browser's address bar. It will look like:

```text
https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz
```

### Run the pipeline

In your activated environment, in the `unlabeled-media-tagger` folder,
run (all on one line — paste it as a single line, even if it wraps in
this document):

```bash
python -m unlabeled_media_tagger "https://drive.google.com/drive/folders/YOUR_FOLDER_ID" --output-dir outputs/pipeline --recursive
```

Replace `YOUR_FOLDER_ID` with your actual folder ID, or just paste the
full URL you copied.

### What you will see

**A browser window opens automatically.** Google asks you to choose an
account — pick the one you added as a test user in step 7d. Then it
will show a screen titled *"Google hasn't verified this app"*.

This warning is normal. The "app" is your own credential that you
created in step 7. The warning exists because you have not gone through
Google's review process — but you do not need to, because you are the
only user.

To proceed:

1. Click **Advanced** (small link, lower left of the warning).
2. Click **Go to Media Tagger (unsafe)** (the wording uses whatever app
   name you gave in step 7d).
3. Click **Continue** to grant the requested Drive permission.

The browser will then say something like *"The authentication flow has
completed. You may close this window."* Close it and return to the
terminal.

**Back in the terminal, the pipeline runs.** You will see:

- Lines about listing files in your Drive folder.
- A long pause and a download bar the **first time only**: DeepFace is
  fetching its face-detection and face-embedding model weights
  (roughly 250 MB). This takes a few minutes on a typical home
  connection. It happens once per machine; later runs reuse the cached
  weights.
- Lines about detecting and embedding faces, then clustering.
- A final summary of how many faces and clusters were found.

When the prompt returns without an error, the run is done.

---

## Step 9 — Where the outputs land

Everything the pipeline produces is in the `outputs/pipeline/` folder
inside the project. Open that folder in your file manager.

The files most useful to you:

- **`share/index.html`** — open this in a web browser. It is a visual
  summary of every cluster the pipeline found, with a *contact sheet*
  (a grid of small face thumbnails) for each one. This is the easiest
  way to eyeball whether the clustering looks right.
- **`share/contact_sheets/`** — the same thumbnails as separate JPEG
  files, one per cluster. Useful if you want to share or annotate them.
- **`share/face_clusters_summary.csv`** — a spreadsheet with one row
  per cluster: the cluster ID, how many faces are in it, and a path to
  its contact sheet. This is the file the review spreadsheet imports.
- **`share/face_clusters_share.csv`** — a spreadsheet with one row per
  detected face, cleaned for sharing.
- **`face_clusters.csv`** — the raw detection data (same format as
  `face_clusters_share.csv` but with full local file paths). Mostly
  useful for debugging; the share copy is what you usually want.

The `share/` folder is rebuilt from scratch on every run. If you keep
running the pipeline, only the latest results are kept there.

---

## Step 10 — Connect to the review spreadsheet

The "review spreadsheet" is the Google Sheet your team uses to look
through clusters, confirm names, and decide which results to write back
to Drive. It pulls in the contact sheets the pipeline produces and
displays them inline using `=IMAGE()` formulas. For that to work, the
images need to live somewhere the spreadsheet can reach — Google Drive.
The pipeline can upload them automatically.

### Create a destination folder in Drive

In Google Drive, create a new folder for the pipeline's uploads. Name
it something obvious like `Media Tagger Outputs` so you can recognize
it later.

> **Important:** put this folder somewhere **outside** the source
> folder you are processing. If you put your upload destination inside
> your source folder, the pipeline's recursive scan will pick up its
> own previous outputs as new media on the next run, which gets messy
> fast. Make the destination folder a sibling of your source folder, or
> in a completely different part of your Drive.

Open the new folder and copy its URL from your browser. It will look
like the source folder URL, just with a different ID at the end.

### Re-run with the upload flag

```bash
python -m unlabeled_media_tagger "https://drive.google.com/drive/folders/YOUR_SOURCE_FOLDER_ID" --output-dir outputs/pipeline --recursive --contact-sheets-drive-folder-id "https://drive.google.com/drive/folders/YOUR_DESTINATION_FOLDER_ID" --cleanup-old-subfolders
```

(Still all on one line. Replace both folder IDs with your actual ones.)

What this changes:

- The contact-sheet JPEGs are uploaded to a timestamped subfolder
  inside your destination folder (named like
  `contact_sheets_2026-04-27_143022`).
- `face_clusters_summary.csv` and `face_clusters_share.csv` are
  uploaded to the top level of your destination folder. Their Drive
  file IDs stay stable across runs, so the review spreadsheet keeps
  working without any reconfiguration.
- The summary CSV's `contact_sheet` column now contains direct Drive
  image URLs the spreadsheet can render with `=IMAGE()`.
- `--cleanup-old-subfolders` deletes earlier auto-named subfolders
  before the new upload, so your destination folder does not collect
  junk over time. Custom-named subfolders are never deleted.

The very first time you use this flag, the OAuth flow may open again
to ask for permission to write to Drive (the previous run only needed
read access). Accept it the same way as in step 8.

After the run, open your review spreadsheet — the new images and CSV
data should be available.

---

## You're done

If you got here with a working `share/index.html` and a populated
review spreadsheet, the pipeline is set up. From here on, day-to-day
use is just:

1. Open your terminal.
2. `cd path/to/unlabeled-media-tagger`
3. `conda activate unlabeled-media-tagger`
4. Run the pipeline command from step 8 or step 10 with whatever Drive
   folder you want to process.

The OAuth login is cached; you will not be prompted again unless you
delete `secrets/token.json` or it expires.

For all other documentation — pipeline internals, configuration
options, developer setup — see the main [README](../README.md).
