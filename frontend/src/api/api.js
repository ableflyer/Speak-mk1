// ============================================================
// api.js — All backend API calls in one place
// Base URL switches between dev and prod automatically
// ============================================================

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:3001";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

// ── AUTH ─────────────────────────────────────────────────────
export const auth = {
  login: (email, password, role) =>
    request("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password, role }),
    }),
  logout: () => request("/api/auth/logout", { method: "POST" }),
};

// ── CHILD ────────────────────────────────────────────────────
export const child = {
  getDashboard: (userId) =>
    request(`/api/child/dashboard?userId=${userId}`),

  getRewards: (userId) =>
    request(`/api/child/rewards?userId=${userId}`),
};

// ── EXERCISE ─────────────────────────────────────────────────
export const exercise = {
  getSession: (userId) =>
    request(`/api/exercise/session?userId=${userId}`),

  // API PLACEHOLDER: audioData should be base64-encoded audio from browser MediaRecorder
  submitRecording: ({ userId, sessionId, exerciseId, audioData, targetPhrase, targetSound, duration }) =>
    request("/api/exercise/submit", {
      method: "POST",
      body: JSON.stringify({ userId, sessionId, exerciseId, audioData, targetPhrase, targetSound, duration }),
    }),

  getResults: (sessionId, userId) =>
    request(`/api/exercise/results?sessionId=${sessionId}&userId=${userId}`),
};

// ── THERAPIST ────────────────────────────────────────────────
export const therapist = {
  getDashboard: (therapistId) =>
    request(`/api/therapist/dashboard?therapistId=${therapistId}`),

  getPatient: (patientId) =>
    request(`/api/therapist/patient/${patientId}`),

  createSession: ({ therapistId, patientId, exerciseIds, notes }) =>
    request("/api/therapist/session/create", {
      method: "POST",
      body: JSON.stringify({ therapistId, patientId, exerciseIds, notes }),
    }),

  getExerciseLibrary: () =>
    request("/api/therapist/exercise-library"),
};

// ── AI (placeholders for backend team) ──────────────────────
export const ai = {
  // API PLACEHOLDER: Backend connects this to LLM of their choice
  getFeedback: ({ transcription, targetPhrase, targetSound, accuracy, childAge, focusArea }) =>
    request("/api/ai/feedback", {
      method: "POST",
      body: JSON.stringify({ transcription, targetPhrase, targetSound, accuracy, childAge, focusArea }),
    }),

  // API PLACEHOLDER: Backend connects this to speech scoring service
  scoreRecording: ({ transcription, targetPhrase, targetSound }) =>
    request("/api/ai/score", {
      method: "POST",
      body: JSON.stringify({ transcription, targetPhrase, targetSound }),
    }),
};
