import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { exercise as exerciseApi } from "../api/api";
import { useApp } from "../context/AppContext";

export default function CheckpointPage() {
  const { user, sessionId, setCurrentExerciseIndex, setSessionResults } = useApp();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const confettiRef = useRef(null);

  useEffect(() => {
    // API: GET /api/exercise/results?sessionId={sid}&userId={uid}
    // API PLACEHOLDER: Backend aggregates all exercise scores for this session
    // API PLACEHOLDER: Backend calls AI for session-level feedback and tip generation
    exerciseApi.getResults(sessionId, user.id)
      .then(d => {
        setData(d);
        launchConfetti();
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [sessionId, user.id]);

  function launchConfetti() {
    if (!confettiRef.current) return;
    const colors = ["#ff9800", "#006b1b", "#00618f", "#91f78e", "#4bbaff"];
    for (let i = 0; i < 40; i++) {
      const el = document.createElement("div");
      el.style.cssText = `
        position: absolute;
        width: ${8 + Math.random() * 8}px;
        height: ${8 + Math.random() * 8}px;
        background: ${colors[Math.floor(Math.random() * colors.length)]};
        border-radius: ${Math.random() > 0.5 ? "50%" : "2px"};
        left: ${Math.random() * 100}%;
        top: -20px;
        animation: confettiFall ${2 + Math.random() * 2}s ${Math.random() * 1}s linear forwards;
        opacity: 0.8;
      `;
      confettiRef.current.appendChild(el);
      setTimeout(() => el.remove(), 4000);
    }
  }

  function handleContinue() {
    setCurrentExerciseIndex(0);
    setSessionResults([]);
    navigate("/");
  }

  if (loading) return (
    <div className="min-h-screen bg-[#fff5ec] flex items-center justify-center">
      <div className="text-center space-y-3">
        <div className="text-6xl animate-bounce">⭐</div>
        <p className="font-headline font-bold text-[#874e00]">Calculating your score...</p>
      </div>
    </div>
  );

  // Progress ring calculation (circumference = 2πr = 2 * π * 88 ≈ 552.9)
  const circumference = 552.92;
  const offset = circumference - (data.accuracy / 100) * circumference;

  return (
    <div className="min-h-screen bg-[#fff5ec] font-body flex flex-col overflow-x-hidden relative">

      {/* Confetti container */}
      <div ref={confettiRef} className="fixed inset-0 pointer-events-none z-10 overflow-hidden" />

      {/* Dot background */}
      <div
        className="fixed inset-0 pointer-events-none z-0 opacity-10"
        style={{
          backgroundImage: "radial-gradient(circle, #ff9800 2px, transparent 2px), radial-gradient(circle, #006b1b 2px, transparent 2px), radial-gradient(circle, #00618f 2px, transparent 2px)",
          backgroundSize: "40px 40px, 60px 60px, 50px 50px",
          backgroundPosition: "0 0, 20px 20px, 10px 40px",
        }}
      />

      {/* ── Header ── */}
      <header className="flex justify-between items-center px-5 py-4 bg-[#fff5ec]/90 backdrop-blur-xl z-50 sticky top-0 border-b border-[#d0a66d]/10">
        <span className="font-headline font-black text-xl text-[#874e00]">Sunny Speech Playground</span>
        <div className="flex gap-2">
          <button className="p-2 rounded-full hover:bg-[#ffeedc] transition-colors text-[#874e00]">
            <span className="material-symbols-outlined">celebration</span>
          </button>
          <button onClick={() => navigate("/")} className="p-2 rounded-full hover:bg-[#ffeedc] transition-colors text-[#874e00]">
            <span className="material-symbols-outlined">home</span>
          </button>
        </div>
      </header>

      <main className="flex-grow flex flex-col items-center justify-center px-5 py-10 relative z-10">
        <div className="w-full max-w-3xl space-y-10">

          {/* ── Hero title ── */}
          <div className="text-center space-y-3">
            <h1 className="font-headline font-extrabold text-5xl md:text-6xl text-[#874e00] tracking-tight -rotate-2 inline-block">
              Checkpoint Complete!
            </h1>
            <p className="text-[#765524] text-xl">You're doing amazing today, Little Explorer!</p>
          </div>

          {/* ── Bento grid: accuracy + stats ── */}
          {/* All data from: GET /api/exercise/results */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-5">

            {/* Accuracy ring */}
            <div className="md:col-span-7 bg-white rounded-2xl p-8 shadow-sm flex flex-col items-center justify-center gap-5 hover:scale-[1.01] transition-transform duration-300">
              <div className="relative">
                <svg className="w-44 h-44 -rotate-90">
                  <circle className="text-[#ffeedc]" cx="88" cy="88" fill="transparent" r="88" stroke="currentColor" strokeWidth="16" />
                  {/* accuracy from GET /api/exercise/results → .accuracy */}
                  <circle
                    className="text-[#006b1b] transition-all duration-1000"
                    cx="88" cy="88" fill="transparent" r="88"
                    stroke="currentColor"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    strokeWidth="16"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="font-headline font-black text-5xl text-[#432900]">{data.accuracy}%</span>
                  <span className="font-bold text-[#006b1b] text-sm tracking-widest uppercase">Accuracy</span>
                </div>
              </div>
              <div className="bg-[#91f78e] text-[#005e17] px-6 py-2 rounded-full font-bold text-lg flex items-center gap-2">
                <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>auto_awesome</span>
                {data.accuracy >= 80 ? "Great job!" : data.accuracy >= 60 ? "Good effort!" : "Keep practicing!"}
              </div>
            </div>

            {/* Stats stack */}
            <div className="md:col-span-5 flex flex-col gap-4">
              {/* Time: from GET /api/exercise/results → .timeTaken */}
              <div className="bg-[#ffeedc] rounded-2xl p-5 flex items-center gap-5">
                <div className="w-14 h-14 bg-[#ffd6a2] rounded-full flex items-center justify-center text-[#874e00]">
                  <span className="material-symbols-outlined text-3xl">schedule</span>
                </div>
                <div>
                  <p className="text-[#765524] text-xs font-bold uppercase tracking-wider">Time Taken</p>
                  <p className="text-[#432900] font-headline font-bold text-3xl">{data.timeTaken}</p>
                </div>
              </div>

              {/* Stars: from GET /api/exercise/results → .starsEarned */}
              <div className="bg-[#ffeedc] rounded-2xl p-5 flex flex-col items-center gap-3 flex-grow">
                <p className="text-[#765524] text-xs font-bold uppercase tracking-wider">Rewards Earned</p>
                <div className="flex gap-1">
                  {[...Array(3)].map((_, i) => (
                    <span
                      key={i}
                      className="material-symbols-outlined drop-shadow-lg"
                      style={{
                        fontSize: i === 1 ? 52 : 40,
                        color: i < data.starsEarned ? "#ff9800" : "#ffd6a2",
                        fontVariationSettings: "'FILL' 1",
                      }}
                    >star</span>
                  ))}
                </div>
                <p className="font-headline font-extrabold text-xl text-[#874e00]">+{data.starsEarned} Stars!</p>
              </div>
            </div>

            {/* Owl Feedback */}
            {/* AI feedback from: POST /api/ai/feedback OR GET /api/exercise/results → .owlFeedback */}
            <div className="md:col-span-12 bg-[#4bbaff]/10 border-2 border-dashed border-[#00618f]/20 rounded-2xl p-7 flex flex-col md:flex-row items-center gap-6">
              <div
                className="w-28 h-28 bg-[#4bbaff] flex items-center justify-center shrink-0 shadow-lg overflow-hidden text-5xl select-none"
                style={{ borderRadius: "65% 35% 72% 28% / 35% 66% 34% 65%", transform: "rotate(3deg)" }}
              >
                🦉
              </div>
              <div className="space-y-2 text-center md:text-left">
                <h3 className="font-headline font-bold text-xl text-[#00618f]">Owl's Wisdom</h3>
                {/* API PLACEHOLDER: owlFeedback generated by AI backend */}
                <p className="text-[#00344f] text-base font-medium leading-relaxed">
                  "{data.owlFeedback}"
                </p>
              </div>
            </div>
          </div>

          {/* ── Exercise breakdown ── */}
          {/* Data from: GET /api/exercise/results → .exerciseBreakdown[] */}
          {data.exerciseBreakdown && (
            <div className="bg-white rounded-2xl p-6 shadow-sm">
              <h3 className="font-headline font-bold text-[#432900] mb-4">Exercise Breakdown</h3>
              <div className="space-y-3">
                {data.exerciseBreakdown.map((item, i) => (
                  <div key={item.id} className="flex items-center justify-between p-3 bg-[#ffeedc] rounded-xl">
                    <div className="flex items-center gap-3">
                      <span className="w-7 h-7 rounded-full bg-[#ff9800] text-white text-xs font-black flex items-center justify-center">{i + 1}</span>
                      <span className="font-medium text-sm text-[#432900] truncate max-w-[160px]">{item.phrase}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-black text-sm" style={{ color: item.accuracy >= 80 ? "#006b1b" : "#b02500" }}>{item.accuracy}%</span>
                      <span className="text-sm">{[...Array(item.stars)].map(() => "⭐").join("")}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Action buttons ── */}
          <div className="flex flex-col sm:flex-row gap-5 justify-center items-center">
            <button
              onClick={() => navigate("/exercise")}
              className="w-full sm:w-auto px-10 py-4 bg-[#ffd6a2] text-[#4a2800] font-headline font-bold text-lg rounded-full hover:scale-105 active:scale-95 transition-all"
            >
              Review Tips
            </button>
            <button
              onClick={handleContinue}
              className="w-full sm:w-auto px-12 py-4 text-white font-headline font-black text-xl rounded-full shadow-xl hover:scale-110 active:scale-95 transition-all flex items-center justify-center gap-3"
              style={{ background: "linear-gradient(135deg, #874e00, #ff9800)" }}
            >
              Continue
              <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>arrow_forward</span>
            </button>
          </div>

        </div>
      </main>

      {/* Bottom Nav */}
      <nav className="fixed bottom-0 left-0 w-full flex justify-around items-center px-4 pb-5 pt-3 bg-white/70 backdrop-blur-xl z-50 rounded-t-3xl shadow-lg border-t border-[#d0a66d]/10">
        {[
          { icon: "videogame_asset", label: "Play", path: "/exercise" },
          { icon: "insert_chart", label: "Progress", path: "/" },
          { icon: "workspace_premium", label: "Rewards", path: "/" },
          { icon: "face", label: "Profile", path: "/" },
        ].map(item => (
          <button
            key={item.label}
            onClick={() => navigate(item.path)}
            className="flex flex-col items-center gap-1 text-[#432900]/60 px-4 py-2 hover:text-[#874e00] transition-all active:scale-90"
          >
            <span className="material-symbols-outlined">{item.icon}</span>
            <span className="text-[11px] font-medium">{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
