import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { child } from "../api/api";
import { useApp } from "../context/AppContext";
import { useTheme } from "../context/ThemeContext";
import BottomNav from "../components/BottomNav";
import ThemePicker from "../components/ThemePicker";

export default function ChildDashboard() {
  const { user } = useApp();
  const { theme } = useTheme();
  const navigate = useNavigate();
  const c = theme.colors;

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showThemePicker, setShowThemePicker] = useState(false);

  useEffect(() => {
    // API: GET /api/child/dashboard?userId={id}
    child.getDashboard(user.id)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [user.id]);

  if (loading) return <LoadingScreen theme={theme} />;

  const accuracyPct = data?.accuracy ?? 0;
  const maxStars = Math.max(...(data?.weeklyProgress ?? [{ stars: 1 }]).map(d => d.stars), 1);

  const heroBgStyle = c.heroBg.startsWith("linear")
    ? { backgroundImage: c.heroBg }
    : { backgroundColor: c.heroBg };

  const heroBtnStyle = c.heroBtnBg.startsWith("linear")
    ? { backgroundImage: c.heroBtnBg, color: "white" }
    : { backgroundColor: c.heroBtnBg, color: theme.id === "speechPlayground" ? "white" : c.accent };

  return (
    <div className="min-h-screen pb-28" style={{ background: c.bg, fontFamily: theme.fonts.body }}>

      {/* ── TopAppBar ── */}
      <header
        className="sticky top-0 z-50 flex justify-between items-center px-5 py-4 border-b shadow-sm"
        style={{ background: c.headerBg, backdropFilter: "blur(20px)", borderColor: `${c.accent}15` }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-11 h-11 rounded-full flex items-center justify-center font-black text-white text-lg border-4 shadow-sm"
            style={{ borderColor: c.starBg, background: `linear-gradient(135deg, ${c.starBg}, ${c.accent})` }}
          >
            {(data?.name ?? "S")[0]}
          </div>
          <div>
            <p className="text-xs font-medium" style={{ color: c.subText }}>Welcome back 👋</p>
            <h1
              className="font-extrabold text-lg leading-none"
              style={{ fontFamily: theme.fonts.headline, color: c.accent }}
            >
              Hi, {data?.name ?? "Sarah"}!
            </h1>
          </div>
        </div>

        <div className="flex gap-2">
          {/* Stars: GET /api/child/rewards → .totalStars */}
          <button
            className="flex items-center gap-1 px-3 py-2 rounded-full font-bold text-sm shadow-sm text-white"
            style={{ background: c.starBg }}
          >
            <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>star</span>
            {data?.totalStars ?? 0}
          </button>

          {/* Theme switcher — opens the picker modal */}
          <button
            onClick={() => setShowThemePicker(true)}
            className="w-10 h-10 flex items-center justify-center rounded-full shadow-sm border transition-all hover:scale-105 active:scale-95"
            style={{ background: "white", borderColor: `${c.accent}20`, color: c.accent }}
            title="Change theme"
          >
            <span className="material-symbols-outlined text-sm">palette</span>
          </button>

          {/* Switch to therapist portal */}
          <button
            onClick={() => navigate("/therapist")}
            className="w-10 h-10 flex items-center justify-center rounded-full shadow-sm border transition-all hover:scale-105"
            style={{ background: "white", borderColor: `${c.accent}20`, color: c.accent }}
            title="Therapist view"
          >
            <span className="material-symbols-outlined text-sm">medical_services</span>
          </button>
        </div>
      </header>

      <main className="px-5 pt-6 max-w-xl mx-auto space-y-5">

        {/* ── Hero Card ── */}
        <section className="relative rounded-2xl p-7 overflow-hidden" style={heroBgStyle}>
          <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full blur-2xl opacity-20" style={{ background: c.accent }} />
          <div className="absolute bottom-0 left-0 w-24 h-24 rounded-full blur-xl opacity-10" style={{ background: c.navActive }} />

          <div className="relative z-10 space-y-3">
            {/* Streak: GET /api/child/dashboard → .streak */}
            <span
              className="inline-flex items-center gap-1 text-white text-xs font-bold px-3 py-1 rounded-full"
              style={{ background: "rgba(0,0,0,0.2)" }}
            >
              <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>
                {theme.streakIcon}
              </span>
              {data?.streak ?? 0} day streak!
            </span>

            <h2
              className="font-extrabold text-2xl leading-tight"
              style={{ fontFamily: theme.fonts.headline, color: c.heroText }}
            >
              {theme.heroTitle}
            </h2>

            <p className="text-sm" style={{ color: c.heroSubText }}>
              {theme.id === "lingoMeadow"
                ? `You've reached ${accuracyPct}% accuracy in your word games this week. Keep going!`
                : `You've reached your daily goal ${data?.streak} days in a row. Keep going, superstar!`}
            </p>

            <button
              onClick={() => navigate("/exercise")}
              className="mt-2 inline-flex items-center gap-2 px-6 py-3 rounded-full font-bold shadow-lg transition-all active:scale-95"
              style={heroBtnStyle}
            >
              <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>play_circle</span>
              {theme.heroBtn}
            </button>
          </div>

          <div className="absolute bottom-4 right-5 text-5xl opacity-20 select-none">{theme.mascot}</div>
        </section>

        {/* ── Progress / Snake section ── */}
        <section className="rounded-2xl p-6 space-y-4" style={{ background: c.snakeBg }}>
          <div className="flex justify-between items-end">
            <div>
              <p className="text-xs font-bold uppercase tracking-widest" style={{ color: c.subText }}>
                {theme.id === "speechPlayground" ? "Accuracy Streak" : "Your Progress"}
              </p>
              <h3 className="font-bold text-lg" style={{ fontFamily: theme.fonts.headline, color: c.primaryText }}>
                {theme.snakeName}
              </h3>
            </div>
            {/* accuracy: GET /api/child/dashboard → .accuracy */}
            <div className="text-right">
              <span className="font-extrabold text-3xl" style={{ color: c.snakeTextPrimary }}>{accuracyPct}%</span>
              <p className="text-xs" style={{ color: c.subText }}>
                {theme.id === "lingoMeadow" ? "+12% vs last week" : "Target Accuracy"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 h-14">
            <div
              className="w-14 h-14 rounded-full flex items-center justify-center shadow-md text-2xl z-10 select-none"
              style={{ background: c.navActive }}
            >
              {theme.mascot}
            </div>
            <div className="flex-1 h-8 rounded-full overflow-hidden" style={{ background: c.snakeTrackBg }}>
              <div
                className="h-full rounded-full transition-all duration-1000"
                style={{ width: `${accuracyPct}%`, background: c.snakeBarBg }}
              />
            </div>
            <div
              className="w-10 h-10 rounded-full flex items-center justify-center text-white shrink-0 shadow"
              style={{ background: c.accent }}
            >
              <span className="material-symbols-outlined text-sm">flag</span>
            </div>
          </div>
          {theme.id === "speechPlayground" && (
            <p className="text-center text-sm italic" style={{ color: c.subText }}>
              Feed {theme.mascotName} more correct sounds to make him grow!
            </p>
          )}
        </section>

        {/* ── Weekly Chart + Badges ── */}
        {/* GET /api/child/dashboard → .weeklyProgress[], .badges[] */}
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2 rounded-2xl p-5 shadow-sm" style={{ background: c.chartBg }}>
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-bold text-sm" style={{ fontFamily: theme.fonts.headline, color: c.primaryText }}>
                Weekly Stars ⭐
              </h3>
              {theme.id === "lingoMeadow" && (
                <span className="text-xs font-bold px-2 py-1 rounded-full flex items-center gap-1" style={{ background: "#eef1f3", color: c.accent }}>
                  <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>trending_up</span>
                  +12%
                </span>
              )}
            </div>
            <div className="flex items-end justify-between h-28 gap-1">
              {(data?.weeklyProgress ?? []).map(({ day, stars }) => {
                const isMax = stars === maxStars;
                const isEmpty = stars === 0;
                return (
                  <div key={day} className="flex flex-col items-center gap-1 w-full">
                    <div
                      className="w-full rounded-t-full transition-all duration-700"
                      style={{
                        height: `${Math.max((stars / maxStars) * 100, isEmpty ? 3 : 8)}%`,
                        background: isMax ? c.chartBarActive : isEmpty ? c.chartBarEmpty : c.chartBarFilled,
                        opacity: isEmpty ? 0.3 : 1,
                      }}
                    />
                    <span className="text-[9px] font-bold" style={{ color: isMax ? c.accent : c.subText }}>{day}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="rounded-2xl p-4 flex flex-col gap-3" style={{ background: c.badgeBg }}>
            <h3 className="font-bold text-xs" style={{ fontFamily: theme.fonts.headline, color: c.badgeText }}>Badges</h3>
            <div className="grid grid-cols-1 gap-2">
              {(data?.badges ?? []).filter(b => b.earned).slice(0, 2).map(badge => (
                <div
                  key={badge.id}
                  className="aspect-square bg-white/40 rounded-xl flex items-center justify-center text-xl hover:scale-110 transition-transform cursor-pointer select-none"
                >
                  {badge.icon}
                </div>
              ))}
            </div>
            <button className="font-bold text-[10px] underline decoration-2 underline-offset-2" style={{ color: c.badgeText }}>View All</button>
          </div>
        </div>

        {/* ── Today's assignment card (ArticPlay + SpeechQuestGreen only) ── */}
        {(theme.id === "articPlay" || theme.id === "speechQuestGreen") && (
          <section
            className="rounded-2xl p-6 flex flex-col md:flex-row items-center gap-6 relative overflow-hidden"
            style={{ background: `${c.accent}12` }}
          >
            <div className="relative z-10 flex-1 space-y-3">
              <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider text-white" style={{ background: c.accent }}>
                Today's Assignment
              </span>
              <h2 className="font-extrabold text-2xl" style={{ fontFamily: theme.fonts.headline, color: c.primaryText }}>
                {theme.heroTitle}
              </h2>
              <p className="text-sm" style={{ color: c.subText }}>
                Help {theme.mascotName} find the missing sounds through 10 fun tongue-twisters!
              </p>
              <button
                onClick={() => navigate("/exercise")}
                className="px-7 py-3 rounded-xl font-bold text-white shadow-lg transition-all active:scale-95 flex items-center gap-2"
                style={{ background: c.accent }}
              >
                Start Exercise
                <span className="material-symbols-outlined text-sm">play_circle</span>
              </button>
            </div>
            {/* Progress ring */}
            <div className="relative w-32 h-32 flex items-center justify-center shrink-0">
              <svg className="w-full h-full -rotate-90">
                <circle cx="64" cy="64" r="54" fill="transparent" stroke={`${c.accent}25`} strokeWidth="10" />
                <circle
                  cx="64" cy="64" r="54" fill="transparent"
                  stroke={c.accent} strokeWidth="10"
                  strokeDasharray="339"
                  strokeDashoffset={339 - (accuracyPct / 100) * 339}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="font-black text-2xl" style={{ color: c.primaryText }}>{accuracyPct}%</span>
                <span className="text-[9px] font-bold uppercase" style={{ color: c.subText }}>Complete</span>
              </div>
            </div>
          </section>
        )}

        {/* ── Goals ── */}
        {/* GET /api/child/dashboard → .goals[] */}
        <section className="space-y-3">
          <h3 className="font-bold" style={{ fontFamily: theme.fonts.headline, color: c.sectionHeading }}>Your Goals</h3>
          {(data?.goals ?? []).map(goal => (
            <div key={goal.id} className="p-5 rounded-2xl flex items-center gap-4" style={{ background: c.goalCardBg }}>
              <div className="w-14 h-14 bg-white rounded-full flex items-center justify-center shadow-sm shrink-0">
                <span className="material-symbols-outlined text-2xl" style={{ fontVariationSettings: "'FILL' 1", color: c.accent }}>
                  {goal.type === "articulation" ? "record_voice_over" : "forum"}
                </span>
              </div>
              <div className="flex-1">
                <div className="flex justify-between mb-1">
                  <span className="font-bold text-sm" style={{ color: c.primaryText }}>{goal.name}</span>
                  <span className="font-bold text-sm" style={{ color: c.accent }}>{goal.current}/{goal.target}</span>
                </div>
                <div className="w-full h-3 rounded-full overflow-hidden" style={{ background: c.goalTrack }}>
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${(goal.current / goal.target) * 100}%`,
                      background: goal.type === "articulation" ? c.goalBar1 : c.goalBar2,
                    }}
                  />
                </div>
              </div>
            </div>
          ))}
        </section>

        {/* ── Recent Activity ── */}
        {/* GET /api/child/dashboard → .recentActivity[] */}
        <section className="space-y-3 pb-4">
          <h3 className="font-bold" style={{ fontFamily: theme.fonts.headline, color: c.sectionHeading }}>Recent Activity</h3>
          {(data?.recentActivity ?? []).map(item => (
            <div key={item.id} className="rounded-2xl p-4 flex items-center justify-between shadow-sm" style={{ background: c.activityBg }}>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: `${c.accent}15` }}>
                  <span className="material-symbols-outlined text-sm" style={{ color: c.accent }}>volume_up</span>
                </div>
                <div>
                  <p className="font-bold text-sm" style={{ color: c.primaryText }}>{item.name}</p>
                  <p className="text-xs" style={{ color: c.subText }}>{item.date}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-black text-sm" style={{ color: item.status === "good" ? c.navActive : "#b02500" }}>
                  {item.accuracy}%
                </span>
                <span
                  className="px-2 py-1 rounded-full text-[10px] font-bold"
                  style={{
                    background: item.status === "good" ? `${c.navActive}20` : "#ffd6a2",
                    color: item.status === "good" ? c.navActive : "#874e00",
                  }}
                >
                  {item.status === "good" ? "Great" : "Needs Work"}
                </span>
              </div>
            </div>
          ))}
        </section>

      </main>

      {/* Theme picker modal */}
      {showThemePicker && <ThemePicker onClose={() => setShowThemePicker(false)} />}

      <BottomNav active="home" role="child" colors={c} fonts={theme.fonts} />
    </div>
  );
}

function LoadingScreen({ theme }) {
  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: theme?.colors?.bg ?? "#fff5ec" }}>
      <div className="text-center space-y-4">
        <div className="text-6xl animate-bounce">{theme?.mascot ?? "🦉"}</div>
        <p className="font-extrabold text-lg" style={{ fontFamily: theme?.fonts?.headline, color: theme?.colors?.accent ?? "#874e00" }}>
          Loading your adventure...
        </p>
      </div>
    </div>
  );
}
