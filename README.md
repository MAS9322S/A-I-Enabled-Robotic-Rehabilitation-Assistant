# AI-Enabled Robotic Rehabilitation Assistant

A **software-only educational rehabilitation prototype** built with Streamlit, MediaPipe, OpenCV and a small Random Forest model. It uses a laptop/desktop webcam to estimate the left elbow angle, count elbow-flexion repetitions, estimate a demonstration form score, and save session history.

> **Important:** This is a college-project prototype, not a medical device. The ML model is trained on synthetic data and is not clinically validated. Do not use its scores or feedback for diagnosis or treatment decisions.

## What is deployment-ready here?

The project is prepared for **Streamlit Community Cloud**:

- Students open one HTTPS URL in Chrome/Edge.
- No Python, VS Code, virtual environment or package installation is required on student laptops.
- Webcam access is requested by the browser.
- WebRTC is configured with public STUN servers for remote connections.
- Supabase can be used as the shared cloud database.
- SQLite remains available as a local-development fallback.
- Secrets are kept outside GitHub.
- Python 3.12 is pinned for a predictable deployment environment.

Streamlit Community Cloud installs Python dependencies from `requirements.txt` and supports choosing the Python version during deployment. See the official deployment documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Project structure

```text
ai_rehab_deployment_ready/
├── app.py
├── requirements.txt
├── runtime.txt
├── packages.txt
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep
├── sql/
│   └── schema.sql
├── utils/
│   ├── __init__.py
│   ├── database.py
│   ├── ml_model.py
│   └── pose_processor.py
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## A. Local test

Use Python 3.12 if possible.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Without Supabase secrets, the app automatically uses a local SQLite file at `data/rehab.db`.

## B. Create the shared cloud database

1. Create a project at https://supabase.com/ .
2. Open **SQL Editor**.
3. Copy all of `sql/schema.sql` into the SQL Editor and run it.
4. Open the project's API/settings page and copy:
   - Project URL
   - `anon` public key

The supplied SQL policies are intentionally simple for a college demonstration and allow anonymous read/write access. **Do not use these policies for real patient/medical data.** For real deployment, add authentication and user-scoped Row Level Security policies.

## C. Upload to GitHub

Create a new GitHub repository, for example:

`ai-rehabilitation-assistant`

Upload the contents of this folder. Do **not** upload `.streamlit/secrets.toml` or any real API credentials.

GitHub: https://github.com/

## D. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/
2. Sign in with GitHub.
3. Click **Create app**.
4. Select your GitHub repository.
5. Select branch `main`.
6. Set the main file to `app.py`.
7. Open **Advanced settings**.
8. Select **Python 3.12**.
9. In **Secrets**, paste:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_ANON_KEY = "YOUR_SUPABASE_ANON_KEY"
```

10. Deploy.

Streamlit's official docs state that secrets should be stored in the Community Cloud Secrets settings rather than committed to GitHub: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

Your final URL will look similar to:

`https://your-project-name.streamlit.app`

You can share that URL with students.

## E. Webcam / WebRTC notes

Remote browser camera access requires HTTPS. `streamlit-webrtc` also requires ICE servers for remote connections. This project includes Google STUN servers by default.

In some restrictive networks, STUN is not enough and a TURN server is required. If that happens, add the following to Streamlit Secrets:

```toml
TURN_URLS = "turn:your-turn-server:3478"
TURN_USERNAME = "your-username"
TURN_CREDENTIAL = "your-password"
```

The application will automatically add the TURN server to its ICE configuration.

Reference: https://github.com/whitphx/streamlit-webrtc

## Student usage

Students only need:

- A modern Chrome/Edge browser
- Internet access
- A working webcam

They do **not** need:

- Python
- VS Code
- Git
- MediaPipe
- OpenCV
- Streamlit
- `.venv`

Workflow:

```text
Student laptop
      │
      ▼
Chrome / Edge
      │
      ▼
https://your-project.streamlit.app
      │
      ▼
Streamlit Community Cloud
      │
      ├── MediaPipe pose detection
      ├── Elbow angle calculation
      ├── Repetition counter
      ├── Demonstration form score
      └── Supabase session storage
```

## Important multi-user limitation

This version uses a shared Supabase database so multiple students can use the same deployed application and see persistent records. The demo policies are open to anonymous users. If your project will handle actual patient information, do not use the demo policies; implement authentication and strict RLS first.

## Troubleshooting

### App fails during dependency installation

Make sure the deployment is using Python 3.12 and the repository contains exactly one dependency file: `requirements.txt`.

### Camera does not connect

1. Confirm the deployed URL starts with `https://`.
2. Allow camera permission in the browser.
3. Try Chrome or Edge.
4. Try a different network/mobile hotspot.
5. If the network blocks peer-to-peer WebRTC traffic, configure a TURN server using the Streamlit Secrets shown above.

### Database error

Check that `SUPABASE_URL` and `SUPABASE_ANON_KEY` are present in Streamlit Cloud Secrets and that `sql/schema.sql` was executed in the correct Supabase project.

### Local development works but cloud deployment fails

Check the Streamlit Cloud logs. Confirm that the repository contains `requirements.txt`, `runtime.txt`, `packages.txt`, `.streamlit/config.toml`, and the `utils/` directory.
