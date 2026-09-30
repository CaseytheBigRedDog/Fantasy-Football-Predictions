# App: API + website + start/sit assistant

Three new pieces that sit on top of the prediction model. None of this changes the model or the public GitHub Pages site.

- `api/` - a FastAPI service that serves the newest `predictions_<season>_week<N>.csv`.
- `web/` - a Next.js website that shows the projections and the assistant.
- Assistant - answers "who should I start?" using only the numbers in your projections.

## How the assistant stays grounded

- Claude never answers from memory. With an `ANTHROPIC_API_KEY` set, it can only get numbers by calling two lookup tools (`lookup_players`, `top_players`) that read your CSV, and it is told not to use outside knowledge about stats or injuries.
- The website shows the exact rows behind every answer.
- With no key, a built-in comparison (no AI) finds the players you named and ranks them by expected points, then explains the floor / ceiling tradeoff on close calls.

## Run it (two PowerShell windows, both from the project folder)

Window 1 - the API:

    cd api
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python test_logic.py
    uvicorn main:app --reload

Window 2 - the website:

    cd web
    npm install
    npm run dev

Open http://localhost:3000. The API's own test page is http://127.0.0.1:8000/docs.

## Turning on the AI answers

    cd api
    copy .env.example .env

Open `api/.env` in a text editor, paste your key after `ANTHROPIC_API_KEY=`, save, and restart uvicorn. The `.env` file is ignored by git, so the key is not uploaded.

## Weekly use

After `python predict_week.py` writes a new predictions file, the API picks it up automatically (no restart needed).

## Not deployed

The public GitHub Pages site is static and cannot run the API or keep an API key secret. This app runs on your computer. Putting it online later needs a host for the API and a host for the website.
