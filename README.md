# SpeechQuest 🦉

A gamified speech therapy app with a child-facing frontend and therapist portal.

---

## Project Structure

```
speechquest/
├── frontend/          ← React.js app (Vite)
│   └── src/
│       ├── pages/
│       │   ├── ChildDashboard.jsx    ← / (Home)
│       │   ├── ExercisePage.jsx      ← /exercise
│       │   ├── AdventureTalk.jsx     ← /adventure
│       │   ├── CheckpointPage.jsx    ← /checkpoint
│       │   └── TherapistPortal.jsx   ← /therapist
│       ├── components/
│       │   └── BottomNav.jsx
│       ├── api/
│       │   └── api.js               ← ALL API calls here
│       └── context/
│           └── AppContext.jsx        ← Global state
└── backend/
    └── server.js                     ← Bun.js server
```

---

## Setup

### Backend (Bun.js)
```bash
cd backend
bun install
bun run dev       # hot-reload
# or
bun run start     # production
```
Runs on: http://localhost:3001

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
Runs on: http://localhost:5173

> Vite proxies all `/api/*` requests to the Bun backend automatically in dev.

---

## Pages & Navigation

| Route         | Page                | User     |
|--------------|---------------------|----------|
| `/`           | Child Dashboard     | Child    |
| `/exercise`   | Recording Exercise  | Child    |
| `/adventure`  | Object ID Exercise  | Child    |
| `/checkpoint` | Results / Stars     | Child    |
| `/therapist`  | Clinical Portal     | Therapist|

---

## API Routes (all in backend/server.js)

| Method | Path                              | Description                          |
|--------|-----------------------------------|--------------------------------------|
| POST   | /api/auth/login                   | Login (child or therapist)           |
| POST   | /api/auth/logout                  | Logout                               |
| GET    | /api/child/dashboard?userId=      | Child home data                      |
| GET    | /api/child/rewards?userId=        | Stars & badges                       |
| GET    | /api/exercise/session?userId=     | Today's exercises list               |
| POST   | /api/exercise/submit              | Submit recording for scoring         |
| GET    | /api/exercise/results?sessionId=  | Session results + AI feedback        |
| GET    | /api/therapist/dashboard          | Therapist home + patient list        |
| GET    | /api/therapist/patient/:id        | Full patient profile + trend         |
| POST   | /api/therapist/session/create     | Create new therapy session           |
| GET    | /api/therapist/exercise-library   | All available exercises              |
| POST   | /api/ai/feedback                  | AI feedback generation (PLACEHOLDER) |
| POST   | /api/ai/score                     | Pronunciation scoring (PLACEHOLDER)  |

---

## API Placeholders (for backend team)

Search for `// API PLACEHOLDER:` in any file to find all integration points.

Key ones:

### `backend/server.js`
- **POST /api/exercise/submit** — connect Speech-to-Text + pronunciation scorer
- **POST /api/ai/feedback** — connect to LLM (OpenAI / Gemini / Claude / etc.)
- **POST /api/ai/score** — connect phoneme scoring service

### `frontend/src/pages/ExercisePage.jsx`
- **`startRecording()`** — MediaRecorder captures audio, convert to base64 for API
- **`handleNext()`** — sends audio to `/api/exercise/submit`

### `frontend/src/pages/AdventureTalk.jsx`
- Camera feed section — connect WebRTC + vision API for mouth tracking

---

## Environment Variables

Create `frontend/.env`:
```
VITE_API_URL=http://localhost:3001
```

For production:
```
VITE_API_URL=https://your-api-domain.com
```
