import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { exercise as exerciseApi, ai } from "../api/api";
import { useApp } from "../context/AppContext";

const RECORDING_SECONDS = 5;

export default function ExercisePage() {
  const { user, sessionId, exercises, setExercises, currentExerciseIndex, setCurrentExerciseIndex, setSessionResults, sessionResults } = useApp();
  const navigate = useNavigate();

  const [currentExercise, setCurrentExercise] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recording, setRecording] = useState(false);
  const [recorded, setRecorded] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [countdown, setCountdown] = useState(RECORDING_SECONDS);
  const [waveBars] = useState(Array.from({ length: 12 }, (_, i) => i));

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    // API: GET /api/exercise/session?userId={id}
    exerciseApi.getSession(user.id)
      .then(data => {
        setExercises(data.exercises);
        setCurrentExercise(data.exercises[currentExerciseIndex]);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [user.id]);

  useEffect(() => {
    if (exercises.length > 0) {
      setCurrentExercise(exercises[currentExerciseIndex]);
      setRecorded(false);
    }
  }, [currentExerciseIndex, exercises]);

  // ── Recording logic ──────────────────────────────────────
  async function startRecording() {
    try {
      // API PLACEHOLDER: Browser MediaRecorder API captures audio
      // API PLACEHOLDER: Audio stream will be sent to /api/exercise/submit as base64
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = e => audioChunksRef.current.push(e.data);
      mediaRecorder.start();
      setRecording(true);
      setCountdown(RECORDING_SECONDS);

      // Countdown timer
      timerRef.current = setInterval(() => {
        setCountdown(prev => {
          if (prev <= 1) {
            clearInterval(timerRef.current);
            stopRecording(mediaRecorder, stream);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } catch (err) {
      // API PLACEHOLDER: Handle microphone permission denied
      alert("Microphone access needed! Please allow microphone access.");
      console.error(err);
    }
  }

  function stopRecording(recorder = mediaRecorderRef.current, stream) {
    if (!recorder) return;
    recorder.stop();
    recorder.onstop = () => {
      if (stream) stream.getTracks().forEach(t => t.stop());
      setRecording(false);
      setRecorded(true);
    };
    clearInterval(timerRef.current);
  }

  // ── Submit recording ─────────────────────────────────────
  async function handleNext() {
    if (!recorded) return;
    setSubmitting(true);

    try {
      // API PLACEHOLDER: Convert audio chunks to base64
      // const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      // const base64 = await blobToBase64(blob);

      // API: POST /api/exercise/submit
      // API PLACEHOLDER: Replace null audioData with actual base64 audio
      const result = await exerciseApi.submitRecording({
        userId: user.id,
        sessionId,
        exerciseId: currentExercise.id,
        audioData: null, // API PLACEHOLDER: replace with base64 audio
        targetPhrase: currentExercise.phrase,
        targetSound: currentExercise.targetSound,
        duration: RECORDING_SECONDS - countdown,
      });

      // API: POST /api/ai/feedback (optional enrichment)
      // API PLACEHOLDER: Backend calls AI to generate child-friendly feedback
      const feedbackRes = await ai.getFeedback({
        transcription: result.transcribed,
        targetPhrase: currentExercise.phrase,
        targetSound: currentExercise.targetSound,
        accuracy: result.accuracy,
        childAge: 6,
        focusArea: "S Sound Articulation",
      });

      setSessionResults(prev => [...prev, { ...result, feedback: feedbackRes.feedback }]);

      // Move to next or finish
      if (currentExerciseIndex + 1 >= exercises.length) {
        navigate("/checkpoint");
      } else {
        setCurrentExerciseIndex(currentExerciseIndex + 1);
        setRecorded(false);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSubmitting(false);
    }
  }

  function handleRetry() {
    setRecorded(false);
    setRecording(false);
    audioChunksRef.current = [];
  }

  if (loading || !currentExercise) return <LoadingExercise />;

  const progressPct = Math.round(((currentExerciseIndex + 1) / exercises.length) * 100);

  return (
    <div className="min-h-screen bg-[#fff5ec] font-body flex flex-col overflow-x-hidden">

      {/* Background blobs */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <div className="absolute top-[-5%] left-[-10%] w-64 h-64 bg-[#ff9800]/10 rounded-full blur-3xl" />
        <div className="absolute bottom-[15%] right-[-5%] w-48 h-48 bg-[#91f78e]/20 rounded-full blur-2xl" />
      </div>

      {/* ── Header + Progress ── */}
      <header className="bg-[#fff5ec]/90 backdrop-blur-xl sticky top-0 z-50 px-5 py-4 border-b border-[#d0a66d]/10">
        <div className="max-w-2xl mx-auto space-y-3">
          <div className="flex items-center justify-between">
            <button
              onClick={() => navigate("/")}
              className="w-9 h-9 flex items-center justify-center rounded-full bg-[#ffeedc] text-[#874e00] hover:bg-[#ffd6a2] transition-all"
            >
              <span className="material-symbols-outlined">arrow_back</span>
            </button>
            <span className="font-headline font-extrabold text-lg text-[#874e00]">Speech Adventure</span>
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-[#ff9800] to-[#874e00] flex items-center justify-center text-white font-black text-sm">
              {user.name[0]}
            </div>
          </div>
          {/* Exercise progress: from GET /api/exercise/session */}
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-xs font-bold text-[#874e00]">
                Exercise {currentExerciseIndex + 1} of {exercises.length}
              </span>
              <span className="text-xs text-[#765524]">{progressPct}% Complete</span>
            </div>
            <div className="h-2.5 w-full bg-[#ffd6a2] rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-[#874e00] to-[#ff9800] rounded-full transition-all duration-700"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        </div>
      </header>

      <main className="flex-grow flex flex-col items-center px-5 py-6 pb-28 max-w-2xl mx-auto w-full gap-6 relative z-10">

        {/* ── Mission card ── */}
        {/* Phrase from: GET /api/exercise/session → exercises[n].phrase */}
        <section className="bg-white rounded-2xl p-7 w-full text-center shadow-sm">
          <p className="text-[#765524] font-bold uppercase tracking-widest text-xs mb-2">Your Mission</p>
          <h1 className="font-headline font-extrabold text-xl text-[#432900] leading-snug">
            {currentExercise.type === "phrase-repeat" ? (
              <>Repeat the phrase:<br/><span className="text-[#874e00] text-2xl">"{currentExercise.phrase}"</span></>
            ) : (
              <>{currentExercise.phrase}</>
            )}
          </h1>
          {currentExercise.instructions && (
            <p className="text-[#765524] text-sm mt-3">{currentExercise.instructions}</p>
          )}
        </section>

        {/* ── Mic / Recording visual ── */}
        <div className="w-full flex flex-col items-center gap-5">
          {/* Big mic button */}
          <div className="relative flex items-center justify-center">
            {recording && (
              <div className="absolute w-32 h-32 rounded-full bg-[#b02500]/20 animate-ping" />
            )}
            <button
              onClick={recording ? () => stopRecording() : startRecording}
              disabled={recorded || submitting}
              className="relative w-28 h-28 rounded-full flex items-center justify-center shadow-2xl transition-all active:scale-95 disabled:opacity-50"
              style={{
                background: recording
                  ? "linear-gradient(135deg, #b02500, #f95630)"
                  : recorded
                  ? "linear-gradient(135deg, #006b1b, #91f78e)"
                  : "linear-gradient(135deg, #874e00, #ff9800)",
              }}
            >
              <span className="material-symbols-outlined text-white text-5xl" style={{ fontVariationSettings: "'FILL' 1" }}>
                {recorded ? "check_circle" : "mic"}
              </span>
              {recording && (
                <span className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-[#b02500] font-black text-lg">{countdown}s</span>
              )}
            </button>
          </div>

          {/* Status text */}
          <p className="text-[#765524] font-medium text-sm text-center">
            {!recording && !recorded && "Tap the mic to start recording"}
            {recording && "Recording... tap to stop early"}
            {recorded && "Great! Tap Next to continue"}
          </p>

          {/* ── Sound wave visualizer ── */}
          {/* API PLACEHOLDER: Wave bars animate to real audio levels when recording */}
          <div className="w-full max-w-sm bg-[#ffeedc] rounded-full h-16 flex items-center justify-center px-6 gap-1">
            {waveBars.map((_, i) => {
              const heights = [16, 32, 20, 40, 28, 44, 18, 36, 24, 42, 20, 30];
              return (
                <div
                  key={i}
                  className="rounded-full transition-all"
                  style={{
                    width: 5,
                    height: recording ? heights[i] : recorded ? 12 : 6,
                    background: recording ? "#006b1b" : recorded ? "#91f78e" : "#d0a66d",
                    animation: recording ? `bounce ${0.6 + i * 0.08}s ease-in-out infinite alternate` : "none",
                  }}
                />
              );
            })}
          </div>

          {/* Encouraging text */}
          {recording && (
            <div className="flex items-center gap-2 px-5 py-2 bg-[#91f78e]/30 rounded-full">
              <span className="material-symbols-outlined text-[#006b1b]">auto_awesome</span>
              <span className="text-[#005d16] font-bold text-sm">Great job! Keep going!</span>
            </div>
          )}
        </div>

      </main>

      {/* ── Bottom Action Bar ── */}
      <nav className="fixed bottom-0 left-0 w-full z-50 flex justify-around items-center px-6 pb-8 pt-4 bg-[#fff5ec]/80 backdrop-blur-xl rounded-t-3xl shadow-lg border-t border-[#d0a66d]/10">
        <button
          onClick={handleRetry}
          disabled={!recorded || submitting}
          className="flex items-center gap-2 bg-[#ffeedc] text-[#874e00] rounded-full px-7 py-3 font-bold text-sm hover:bg-[#ffd6a2] transition-all active:scale-95 disabled:opacity-40"
        >
          <span className="material-symbols-outlined">refresh</span>
          Retry
        </button>
        <button
          onClick={handleNext}
          disabled={!recorded || submitting}
          className="flex items-center gap-2 text-white rounded-full px-8 py-3 font-bold text-sm transition-all active:scale-95 disabled:opacity-40 shadow-lg"
          style={{ background: "linear-gradient(135deg, #874e00, #ff9800)" }}
        >
          {submitting ? "Scoring..." : currentExerciseIndex + 1 >= exercises.length ? "Finish" : "Next"}
          <span className="material-symbols-outlined">arrow_forward</span>
        </button>
      </nav>
    </div>
  );
}

function LoadingExercise() {
  return (
    <div className="min-h-screen bg-[#fff5ec] flex items-center justify-center">
      <div className="text-center space-y-3">
        <div className="text-5xl animate-spin">⭐</div>
        <p className="font-headline font-bold text-[#874e00]">Preparing your exercise...</p>
      </div>
    </div>
  );
}
