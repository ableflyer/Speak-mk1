// ============================================================
// SpeechQuest Backend — Bun.js Server
// Run: bun run server.js
// ============================================================

const PORT = 3001;

// ── CORS helper ──────────────────────────────────────────────
const cors = {
  "Access-Control-Allow-Origin": "http://localhost:5173",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...cors },
  });
}

function error(message, status = 400) {
  return json({ error: message }, status);
}

// ── Route Handler ────────────────────────────────────────────
async function handleRequest(req) {
  const url = new URL(req.url);
  const path = url.pathname;
  const method = req.method;

  // Preflight
  if (method === "OPTIONS") return new Response(null, { headers: cors });

  // ── AUTH ─────────────────────────────────────────────────
  // POST /api/auth/login
  if (path === "/api/auth/login" && method === "POST") {
    const body = await req.json();
    // API PLACEHOLDER: Verify credentials against database
    // API PLACEHOLDER: Generate JWT token
    // Expected body: { email, password, role: "child" | "therapist" }
    return json({
      token: "placeholder-jwt-token",
      user: {
        id: "user-001",
        name: body.role === "therapist" ? "Dr. Sarah Jenkins" : "Sarah",
        role: body.role,
        avatarUrl: null,
      },
    });
  }

  // POST /api/auth/logout
  if (path === "/api/auth/logout" && method === "POST") {
    // API PLACEHOLDER: Invalidate JWT token / session
    return json({ success: true });
  }

  // ── CHILD DASHBOARD ──────────────────────────────────────
  // GET /api/child/dashboard
  if (path === "/api/child/dashboard" && method === "GET") {
    const userId = url.searchParams.get("userId");
    // API PLACEHOLDER: Fetch child profile, streak, accuracy from database
    return json({
      userId,
      name: "Sarah",
      avatarUrl: null,
      streak: 3,
      totalStars: 47,
      accuracy: 85,
      dailyGoalMet: true,
      weeklyProgress: [
        { day: "Mon", stars: 40 },
        { day: "Tue", stars: 65 },
        { day: "Wed", stars: 85 },
        { day: "Thu", stars: 100 },
        { day: "Fri", stars: 20 },
        { day: "Sat", stars: 15 },
        { day: "Sun", stars: 0 },
      ],
      goals: [
        { id: "g1", name: "Articulation 'S'", current: 12, target: 15, type: "articulation", color: "primary" },
        { id: "g2", name: "Sentence Building", current: 4, target: 10, type: "sentence", color: "tertiary" },
      ],
      badges: [
        { id: "b1", name: "Trophy", icon: "🏆", earned: true },
        { id: "b2", name: "Streak Fire", icon: "🔥", earned: true },
        { id: "b3", name: "Star", icon: "⭐", earned: false },
        { id: "b4", name: "Diamond", icon: "💎", earned: false },
      ],
      recentActivity: [
        { id: "a1", name: "Sound Exercise: S Initial", date: "Today, 2:30 PM", accuracy: 92, status: "good" },
        { id: "a2", name: "Picture Association", date: "Yesterday, 4:15 PM", accuracy: 88, status: "good" },
        { id: "a3", name: "Sentence Echo", date: "Yesterday, 10:00 AM", accuracy: 64, status: "needs-work" },
      ],
    });
  }

  // ── EXERCISE SESSION ─────────────────────────────────────
  // GET /api/exercise/session
  if (path === "/api/exercise/session" && method === "GET") {
    const userId = url.searchParams.get("userId");
    // API PLACEHOLDER: Fetch today's assigned exercises for this child from database
    // API PLACEHOLDER: Apply therapist-configured exercise plan
    return json({
      sessionId: "sess-001",
      userId,
      totalExercises: 5,
      exercises: [
        {
          id: "ex-001",
          exerciseNumber: 1,
          type: "phrase-repeat",
          phrase: "The snake slithers slowly.",
          instructions: "Repeat the phrase clearly into the microphone.",
          imageUrl: null,
          targetSound: "S",
          difficulty: "medium",
        },
        {
          id: "ex-002",
          exerciseNumber: 2,
          type: "object-identification",
          phrase: "How do you say pizza?",
          imageUrl: null,
          targetWord: "pizza",
          targetSound: "P",
          difficulty: "easy",
        },
        {
          id: "ex-003",
          exerciseNumber: 3,
          type: "phrase-repeat",
          phrase: "Sally sells seashells by the seashore.",
          instructions: "Say this tongue-twister as clearly as you can!",
          imageUrl: null,
          targetSound: "S",
          difficulty: "hard",
        },
        {
          id: "ex-004",
          exerciseNumber: 4,
          type: "phrase-repeat",
          phrase: "The big brown bear.",
          instructions: "Repeat the phrase clearly.",
          imageUrl: null,
          targetSound: "B",
          difficulty: "easy",
        },
        {
          id: "ex-005",
          exerciseNumber: 5,
          type: "object-identification",
          phrase: "What animal is this?",
          imageUrl: null,
          targetWord: "rabbit",
          targetSound: "R",
          difficulty: "medium",
        },
      ],
    });
  }

  // POST /api/exercise/submit
  if (path === "/api/exercise/submit" && method === "POST") {
    const body = await req.json();
    // API PLACEHOLDER: Receive audio recording (base64 or blob URL)
    // API PLACEHOLDER: Send audio to Speech-to-Text service (e.g. Google STT, Azure, Whisper)
    //   const transcription = await speechToText(body.audioData)
    // API PLACEHOLDER: Score the transcription against the target phrase
    //   const score = await scorePronunciation(transcription, body.targetPhrase)
    // API PLACEHOLDER: Generate AI feedback using LLM
    //   const feedback = await generateFeedback(score, body.targetSound)
    // API PLACEHOLDER: Save result to database
    //   await db.exerciseResults.create({ userId, exerciseId, score, accuracy, ... })
    // Expected body: { userId, sessionId, exerciseId, audioData, targetPhrase, targetSound, duration }
    return json({
      exerciseId: body.exerciseId,
      score: 85,
      accuracy: 85,
      transcribed: "The snake slithers slowly",
      targetPhrase: body.targetPhrase,
      feedback: "Great job! Watch your 'S' sound placement — try to keep your tongue behind your teeth!",
      stars: 3,
      passed: true,
      nextExerciseId: "ex-002",
    });
  }

  // ── CHECKPOINT RESULTS ───────────────────────────────────
  // GET /api/exercise/results
  if (path === "/api/exercise/results" && method === "GET") {
    const sessionId = url.searchParams.get("sessionId");
    const userId = url.searchParams.get("userId");
    // API PLACEHOLDER: Aggregate all exercise results for this session from database
    // API PLACEHOLDER: Calculate total accuracy, stars earned, time taken
    // API PLACEHOLDER: Generate session-level AI feedback
    return json({
      sessionId,
      userId,
      accuracy: 85,
      timeTaken: "3:20",
      starsEarned: 3,
      totalStars: 50,
      passed: true,
      owlFeedback: "Good effort! You're getting so much better. Watch your 'S' sound placement next time — try to keep your tongue behind your teeth!",
      newBadges: [],
      exerciseBreakdown: [
        { id: "ex-001", phrase: "The snake slithers slowly.", accuracy: 85, stars: 3 },
        { id: "ex-002", phrase: "Pizza", accuracy: 90, stars: 3 },
        { id: "ex-003", phrase: "Sally sells seashells...", accuracy: 72, stars: 2 },
        { id: "ex-004", phrase: "The big brown bear.", accuracy: 88, stars: 3 },
        { id: "ex-005", phrase: "Rabbit", accuracy: 80, stars: 3 },
      ],
    });
  }

  // ── THERAPIST PORTAL ─────────────────────────────────────
  // GET /api/therapist/dashboard
  if (path === "/api/therapist/dashboard" && method === "GET") {
    const therapistId = url.searchParams.get("therapistId");
    // API PLACEHOLDER: Fetch therapist info and today's sessions from database
    return json({
      therapistId,
      name: "Dr. Sarah Jenkins",
      title: "Speech Pathologist",
      avatarUrl: null,
      todaySessions: 4,
      dailyGoalsMet: 12,
      dailyGoalsTarget: 15,
      nextPatient: "Sarah G.",
      patients: [
        {
          id: "p-001",
          name: "Sarah G.",
          age: 6,
          status: "active",
          avatarUrl: null,
          focusArea: "Phonological Awareness & 'S' Sound Articulation",
          sessionsCompleted: 14,
          totalSessions: 20,
          avgAccuracy: 82,
          currentLevel: "Intermediate",
        },
        {
          id: "p-002",
          name: "Leo M.",
          age: 7,
          status: "active",
          avatarUrl: null,
          focusArea: "Vowel Sounds & Sentence Building",
          sessionsCompleted: 8,
          totalSessions: 20,
          avgAccuracy: 74,
          currentLevel: "Beginner",
        },
      ],
    });
  }

  // GET /api/therapist/patient/:patientId
  if (path.startsWith("/api/therapist/patient/") && method === "GET") {
    const patientId = path.split("/")[4];
    // API PLACEHOLDER: Fetch full patient profile, all sessions, progress data
    return json({
      id: patientId,
      name: "Sarah G.",
      age: 6,
      focusArea: "Phonological Awareness & 'S' Sound Articulation",
      sessionsCompleted: 14,
      totalSessions: 20,
      avgAccuracy: 82,
      exercisesCompleted: 85,
      totalExercises: 100,
      currentLevel: 4,
      maxLevel: 5,
      perfectScores: 3,
      weeklyTrend: [
        { session: 1, accuracy: 65 },
        { session: 2, accuracy: 70 },
        { session: 3, accuracy: 75 },
        { session: 4, accuracy: 68 },
        { session: 5, accuracy: 82 },
        { session: 6, accuracy: 85 },
        { session: 7, accuracy: 86 },
      ],
      milestones: [
        { name: "80% Accuracy Average", target: 80, current: 82, met: true },
        { name: "15 Sessions Completed", target: 15, current: 14, met: false },
      ],
      recentSessions: [
        { date: "Oct 26, 2023", exercise: "Sound Exercise: S Initial", accuracy: 85, status: "target-met" },
        { date: "Oct 25, 2023", exercise: "Vowel Challenge: Long A", accuracy: 78, status: "near-target" },
        { date: "Oct 23, 2023", exercise: "Word Blending: Compound", accuracy: 82, status: "target-met" },
      ],
    });
  }

  // POST /api/therapist/session/create
  if (path === "/api/therapist/session/create" && method === "POST") {
    const body = await req.json();
    // API PLACEHOLDER: Create a new therapy session in database
    // API PLACEHOLDER: Assign exercises based on therapist configuration
    // Expected body: { therapistId, patientId, exerciseIds[], notes }
    return json({ sessionId: "sess-new-001", created: true });
  }

  // GET /api/therapist/exercise-library
  if (path === "/api/therapist/exercise-library" && method === "GET") {
    // API PLACEHOLDER: Fetch all available exercises from database
    // API PLACEHOLDER: Filter by sound, difficulty, type
    return json({
      exercises: [
        { id: "lib-001", name: "S Sound Basics", targetSound: "S", difficulty: "easy", type: "phrase-repeat" },
        { id: "lib-002", name: "Pizza Object ID", targetSound: "P", difficulty: "easy", type: "object-identification" },
        { id: "lib-003", name: "Tongue Twisters - S", targetSound: "S", difficulty: "hard", type: "phrase-repeat" },
      ],
    });
  }

  // ── AI FEEDBACK (direct endpoint) ───────────────────────
  // POST /api/ai/feedback
  if (path === "/api/ai/feedback" && method === "POST") {
    const body = await req.json();
    // API PLACEHOLDER: Send to LLM (OpenAI / Gemini / Claude / etc.)
    // API PLACEHOLDER: Prompt engineering for child-friendly, encouraging feedback
    // API PLACEHOLDER: Include therapist-defined focus areas in prompt context
    // Expected body: { transcription, targetPhrase, targetSound, accuracy, childAge, focusArea }
    return json({
      feedback: "Great effort! Keep your tongue behind your teeth for the 'S' sound. You're getting so much better!",
      tips: ["Keep tongue behind teeth", "Take a deep breath before starting"],
      encouragement: "You're a superstar! 🌟",
    });
  }

  // POST /api/ai/score
  if (path === "/api/ai/score" && method === "POST") {
    const body = await req.json();
    // API PLACEHOLDER: Receive transcribed text and target phrase
    // API PLACEHOLDER: Run pronunciation scoring algorithm or send to AI scoring service
    // API PLACEHOLDER: Return phoneme-level breakdown if available
    // Expected body: { transcription, targetPhrase, targetSound }
    return json({
      overallScore: 85,
      soundAccuracy: { S: 80, th: 90, sl: 85 },
      fluency: 88,
      passed: true,
    });
  }

  // ── REWARDS ──────────────────────────────────────────────
  // GET /api/child/rewards
  if (path === "/api/child/rewards" && method === "GET") {
    const userId = url.searchParams.get("userId");
    // API PLACEHOLDER: Fetch all rewards, badges, star balance from database
    return json({
      userId,
      totalStars: 47,
      badges: [
        { id: "b1", name: "Trophy", icon: "🏆", earned: true, earnedDate: "2024-01-10" },
        { id: "b2", name: "Streak Fire", icon: "🔥", earned: true, earnedDate: "2024-01-12" },
        { id: "b3", name: "Star Master", icon: "⭐", earned: false },
        { id: "b4", name: "Diamond", icon: "💎", earned: false },
        { id: "b5", name: "Word Master", icon: "📚", earned: false },
        { id: "b6", name: "Vowel King", icon: "👑", earned: false },
      ],
    });
  }

  // 404
  return json({ error: "Route not found" }, 404);
}

// ── Start Server ─────────────────────────────────────────────
const server = Bun.serve({
  port: PORT,
  fetch: handleRequest,
});

console.log(`✅ SpeechQuest backend running on http://localhost:${PORT}`);
console.log(`\n📋 Available API Routes:`);
console.log(`   POST /api/auth/login`);
console.log(`   POST /api/auth/logout`);
console.log(`   GET  /api/child/dashboard?userId=`);
console.log(`   GET  /api/child/rewards?userId=`);
console.log(`   GET  /api/exercise/session?userId=`);
console.log(`   POST /api/exercise/submit`);
console.log(`   GET  /api/exercise/results?sessionId=&userId=`);
console.log(`   GET  /api/therapist/dashboard?therapistId=`);
console.log(`   GET  /api/therapist/patient/:id`);
console.log(`   POST /api/therapist/session/create`);
console.log(`   GET  /api/therapist/exercise-library`);
console.log(`   POST /api/ai/feedback`);
console.log(`   POST /api/ai/score`);
