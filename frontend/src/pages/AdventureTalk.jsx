import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { exercise as exerciseApi } from "../api/api";
import { useApp } from "../context/AppContext";
import BottomNav from "../components/BottomNav";

export default function AdventureTalk() {
  const { user, sessionId } = useApp();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // API: GET /api/exercise/session?userId={id}
    // Picks the object-identification type exercise
    exerciseApi.getSession(user.id)
      .then(d => {
        const objEx = d.exercises.find(e => e.type === "object-identification") || d.exercises[0];
        setData({ ...d, currentExercise: objEx });
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [user.id]);

  if (loading) return (
    <div className="min-h-screen bg-[#fff5ec] flex items-center justify-center">
      <div className="text-5xl animate-bounce">🍕</div>
    </div>
  );

  const ex = data?.currentExercise;

  return (
    <div className="min-h-screen bg-[#fff5ec] font-body">

      {/* Floating decor */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <div className="absolute top-20 left-10 text-[#4bbaff]/30 select-none animate-pulse">
          <span className="material-symbols-outlined text-6xl">auto_awesome</span>
        </div>
        <div className="absolute bottom-40 right-10 text-[#ff9800]/20 select-none rotate-12">
          <span className="material-symbols-outlined text-8xl">bubble_chart</span>
        </div>
      </div>

      {/* ── TopNav ── */}
      <nav className="bg-[#fff5ec] sticky top-0 z-50 flex justify-between items-center w-full px-5 py-4 max-w-7xl mx-auto border-b border-[#d0a66d]/10">
        <div className="text-2xl font-black text-[#874e00] font-headline">Adventure Talk</div>

        {/* Adventure Map Progress: GET /api/exercise/session → progress */}
        <div className="hidden md:flex flex-1 mx-10 items-center gap-4 bg-[#ffeedc] px-5 py-3 rounded-full relative overflow-hidden">
          <span className="material-symbols-outlined text-[#874e00]" style={{ fontVariationSettings: "'FILL' 1" }}>flag</span>
          <div className="flex-1 h-3 bg-[#ffd6a2] rounded-full overflow-hidden relative">
            <div className="h-full bg-[#006b1b] rounded-full" style={{ width: "65%" }}>
              <div className="absolute right-0 top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full border-2 border-[#006b1b] scale-125" />
            </div>
          </div>
          <span className="font-headline font-bold text-[#874e00] text-sm whitespace-nowrap">Level 4: Whispering Woods</span>
        </div>

        <div className="flex items-center gap-3">
          <button onClick={() => navigate("/")} className="p-2 rounded-full hover:bg-[#ffeedc] transition-colors text-[#874e00]">
            <span className="material-symbols-outlined">home</span>
          </button>
          <div className="w-10 h-10 rounded-full bg-[#ff9800] flex items-center justify-center text-white font-black text-sm">
            {user.name[0]}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-5 pt-6 pb-32 grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">

        {/* ── Left Column: Camera + Character Feedback ── */}
        <div className="lg:col-span-4 flex flex-col gap-5">

          {/* Camera view */}
          {/* API PLACEHOLDER: Camera feed for real-time face/mouth tracking */}
          {/* API PLACEHOLDER: WebRTC stream → backend vision API for mouth position analysis */}
          <div className="bg-white rounded-2xl p-4 shadow-sm overflow-hidden border-4 border-[#ffeedc] relative">
            <div className="aspect-square rounded-xl bg-[#ffeedc] overflow-hidden relative flex items-center justify-center">
              <div className="text-center space-y-2">
                <span className="material-symbols-outlined text-6xl text-[#d0a66d]">videocam</span>
                <p className="text-[#765524] text-sm font-medium">Camera feed here</p>
                <p className="text-[#95703c] text-xs">
                  {/* API PLACEHOLDER: Request camera permission + stream */}
                  Connect camera for mouth tracking
                </p>
              </div>
              {/* Live tracking badge */}
              <div className="absolute top-3 left-3 bg-[#006b1b]/90 text-white px-3 py-1 rounded-full text-xs font-bold flex items-center gap-1">
                <span className="w-2 h-2 bg-white rounded-full animate-ping" />
                Live Tracking
              </div>
            </div>
            <div className="mt-4 flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-[#91f78e] flex items-center justify-center">
                <span className="material-symbols-outlined text-[#005e17]">face</span>
              </div>
              <div>
                {/* API PLACEHOLDER: Mouth position from vision model */}
                <div className="font-headline font-bold text-[#432900] text-sm">Mouth Position</div>
                <div className="text-xs text-[#765524]">Perfectly Aligned!</div>
              </div>
            </div>
          </div>

          {/* Character Feedback */}
          {/* API PLACEHOLDER: Feedback text from /api/ai/feedback after submission */}
          <div className="bg-[#4bbaff] rounded-2xl p-7 relative overflow-hidden group" style={{ borderRadius: "65% 35% 72% 28% / 35% 66% 34% 65%" }}>
            <div className="absolute -top-4 -right-4 text-white/20 rotate-12 group-hover:rotate-45 transition-transform duration-700">
              <span className="material-symbols-outlined text-9xl">pets</span>
            </div>
            <div className="relative z-10">
              <h3 className="font-headline text-2xl font-black text-[#00344f] mb-2">Super job!</h3>
              <p className="text-[#00344f]/90 text-base leading-relaxed">
                "I love how you said that '{ex?.targetSound}' sound! Can you say it one more time?"
              </p>
            </div>
          </div>
        </div>

        {/* ── Right Column: Exercise ── */}
        <div className="lg:col-span-8 space-y-6">

          {/* Exercise header */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <span className="inline-block px-4 py-1 rounded-full bg-[#ff9800] text-[#4a2800] font-bold text-xs mb-2 uppercase tracking-wider">
                {ex?.type === "object-identification" ? "Object Identification" : "Phrase Repeat"}
              </span>
              {/* Phrase from: GET /api/exercise/session → exercise.phrase */}
              <h1 className="text-3xl font-headline font-black text-[#874e00]">{ex?.phrase}</h1>
            </div>
            <div className="flex gap-2">
              {/* API PLACEHOLDER: Play audio pronunciation of the target word */}
              <button className="w-12 h-12 rounded-full bg-[#ffeedc] flex items-center justify-center text-[#874e00] hover:bg-[#ffd6a2] transition-colors">
                <span className="material-symbols-outlined">volume_up</span>
              </button>
              <button className="w-12 h-12 rounded-full bg-[#ffeedc] flex items-center justify-center text-[#874e00] hover:bg-[#ffd6a2] transition-colors">
                <span className="material-symbols-outlined">lightbulb</span>
              </button>
            </div>
          </div>

          {/* Main activity area */}
          <div className="bg-white rounded-2xl p-8 md:p-12 shadow-sm min-h-80 flex flex-col items-center justify-center border-b-8 border-[#ffeedc]">

            {/* Target image */}
            {/* API PLACEHOLDER: Image URL from exercise data: GET /api/exercise/session → exercise.imageUrl */}
            <div
              className="w-56 h-56 bg-[#ffeedc] flex items-center justify-center p-6 mb-8 cursor-pointer hover:scale-105 transition-transform duration-300 text-8xl select-none"
              style={{ borderRadius: "30% 70% 70% 30% / 50% 60% 40% 50%" }}
            >
              {ex?.targetWord === "pizza" ? "🍕" : ex?.targetWord === "rabbit" ? "🐰" : "🎯"}
            </div>

            {/* Sound wave */}
            {/* API PLACEHOLDER: Wave animates to actual mic input levels */}
            <div className="w-full max-w-md bg-[#ffeedc] rounded-full h-16 flex items-center justify-center px-8 gap-1 mb-6">
              {[16, 32, 48, 24, 40, 16, 32, 48, 24, 40, 16].map((h, i) => (
                <div
                  key={i}
                  className="rounded-full bg-[#006b1b]"
                  style={{
                    width: 4,
                    height: h,
                    animation: `pulse ${0.8 + i * 0.1}s ease-in-out infinite alternate`,
                  }}
                />
              ))}
            </div>

            {/* Target word display */}
            {ex?.targetWord && (
              <div className="text-2xl font-headline font-black text-[#006b1b] tracking-widest uppercase">
                {ex.targetWord.split("").join(" ").toUpperCase()} !
              </div>
            )}
          </div>

          {/* Next quests preview */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-[#ffeedc] p-5 rounded-2xl flex items-center gap-4 border-2 border-transparent hover:border-[#874e00]/20 transition-all cursor-pointer">
              <div className="w-14 h-14 rounded-xl bg-white flex items-center justify-center shadow-sm">
                <span className="material-symbols-outlined text-3xl text-[#874e00]">auto_stories</span>
              </div>
              <div>
                <h4 className="font-headline font-bold text-[#432900] text-sm">Next Quest: Story</h4>
                <p className="text-xs text-[#765524]">The Brave Little Lion</p>
              </div>
            </div>
            <div className="bg-[#ffeedc] p-5 rounded-2xl flex items-center gap-4 border-2 border-transparent hover:border-[#006b1b]/20 transition-all cursor-pointer">
              <div className="w-14 h-14 rounded-xl bg-white flex items-center justify-center shadow-sm">
                <span className="material-symbols-outlined text-3xl text-[#006b1b]">extension</span>
              </div>
              <div>
                <h4 className="font-headline font-bold text-[#432900] text-sm">Sound Puzzle</h4>
                <p className="text-xs text-[#765524]">Match the 'Sss' sound</p>
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex flex-col sm:flex-row gap-4">
            <button
              onClick={() => navigate("/exercise")}
              className="flex-1 bg-[#ffd6a2] text-[#4a2800] font-headline font-black py-5 px-8 rounded-full text-lg hover:bg-[#ffeedc] transition-all flex items-center justify-center gap-3 active:scale-95"
            >
              <span className="material-symbols-outlined">replay</span>
              Retry
            </button>
            <button
              onClick={() => navigate("/checkpoint")}
              className="flex-[2] text-white font-headline font-black py-5 px-8 rounded-full text-lg shadow-lg hover:opacity-90 transition-all flex items-center justify-center gap-3 active:scale-95"
              style={{ background: "linear-gradient(135deg, #874e00, #ff9800)" }}
            >
              Next Adventure
              <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>arrow_forward</span>
            </button>
          </div>
        </div>
      </main>

      <BottomNav active="games" role="child" />
    </div>
  );
}
