import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { therapist as therapistApi } from "../api/api";
import { useApp } from "../context/AppContext";

const THERAPIST_ID = "therapist-001";

export default function TherapistPortal() {
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [patient, setPatient] = useState(null);
  const [activeNav, setActiveNav] = useState("patients");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // API: GET /api/therapist/dashboard?therapistId={id}
    therapistApi.getDashboard(THERAPIST_ID)
      .then(d => {
        setData(d);
        if (d.patients?.[0]) loadPatient(d.patients[0].id);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  async function loadPatient(patientId) {
    // API: GET /api/therapist/patient/:patientId
    try {
      const p = await therapistApi.getPatient(patientId);
      setPatient(p);
    } catch (e) { console.error(e); }
  }

  if (loading) return (
    <div className="min-h-screen bg-[#f4f6ff] flex items-center justify-center">
      <div className="text-center space-y-3">
        <div className="text-5xl animate-pulse">🩺</div>
        <p className="font-headline font-bold text-[#874e00]">Loading portal...</p>
      </div>
    </div>
  );

  const navItems = [
    { id: "overview", label: "Overview", icon: "dashboard" },
    { id: "patients", label: "Patient Records", icon: "group" },
    { id: "library", label: "Exercise Library", icon: "library_books" },
    { id: "reports", label: "Reports", icon: "assessment" },
    { id: "settings", label: "Settings", icon: "settings" },
  ];

  const maxAccuracy = Math.max(...(patient?.weeklyTrend ?? [{ accuracy: 0 }]).map(d => d.accuracy));

  return (
    <div className="min-h-screen bg-[#f4f6ff] font-body text-[#212f42] flex h-screen overflow-hidden">

      {/* ── Sidebar ── */}
      <aside className="hidden md:flex flex-col h-full w-64 bg-[#eaf1ff] p-4 gap-2 rounded-r-3xl shrink-0">
        <div className="px-4 py-5 mb-2 flex items-center gap-3">
          <div className="w-10 h-10 bg-[#ff9800] rounded-full flex items-center justify-center text-white">
            <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>record_voice_over</span>
          </div>
          <div>
            <h1 className="text-base font-extrabold text-[#874e00]">SpeechQuest</h1>
            <p className="text-xs text-[#4e5c71]">Clinical Portal</p>
          </div>
        </div>

        <nav className="flex flex-col gap-1">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveNav(item.id)}
              className="flex items-center gap-3 px-4 py-3 rounded-full text-sm font-medium transition-all duration-200"
              style={{
                background: activeNav === item.id ? "white" : "transparent",
                color: activeNav === item.id ? "#874e00" : "#4e5c71",
                boxShadow: activeNav === item.id ? "0 2px 8px rgba(33,47,66,0.06)" : "none",
              }}
            >
              <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: activeNav === item.id ? "'FILL' 1" : "'FILL' 0" }}>
                {item.icon}
              </span>
              {item.label}
            </button>
          ))}
        </nav>

        <div className="mt-auto p-2">
          {/* API: POST /api/therapist/session/create */}
          <button className="w-full bg-[#ff9800] text-white py-4 rounded-full font-bold shadow-sm active:scale-95 transition-all flex items-center justify-center gap-2">
            <span className="material-symbols-outlined">add_circle</span>
            New Session
          </button>
        </div>
      </aside>

      {/* ── Main viewport ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* TopBar */}
        <header className="flex justify-between items-center w-full px-6 py-3 h-16 bg-[#f4f6ff] shadow-sm z-10">
          <div className="flex items-center gap-4">
            <div className="md:hidden text-lg font-black text-[#874e00]">SpeechQuest</div>
            <div className="hidden md:flex items-center bg-[#eaf1ff] px-4 py-2 rounded-full w-80">
              <span className="material-symbols-outlined text-[#6a788d] text-sm">search</span>
              {/* API PLACEHOLDER: Search patients by name / session */}
              <input
                className="bg-transparent border-none outline-none text-sm w-full ml-2 placeholder:text-[#6a788d]"
                placeholder="Search patients or sessions..."
              />
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button className="p-2 rounded-full hover:bg-white/50 relative">
              <span className="material-symbols-outlined text-[#4e5c71]">notifications</span>
              <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full" />
            </button>
            <div className="flex items-center gap-3 pl-4 border-l border-[#a0aec5]/20">
              <div className="text-right hidden sm:block">
                {/* from GET /api/therapist/dashboard → .name */}
                <p className="text-xs font-bold text-[#212f42]">{data?.name}</p>
                <p className="text-[10px] text-[#4e5c71]">{data?.title}</p>
              </div>
              <div className="w-10 h-10 rounded-full bg-[#ff9800] flex items-center justify-center text-white font-black text-sm">
                {(data?.name ?? "D")[2]}
              </div>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-6 space-y-7 bg-[#f4f6ff]">

          {/* Welcome */}
          <section className="flex flex-col md:flex-row md:items-end justify-between gap-4">
            <div>
              {/* from GET /api/therapist/dashboard → .name, .nextPatient, .todaySessions */}
              <h2 className="text-2xl font-extrabold text-[#212f42] tracking-tight">Good Morning, {data?.name?.split(" ")[0]} {data?.name?.split(" ")[1]}</h2>
              <p className="text-[#4e5c71] mt-1">
                You have {data?.todaySessions} sessions scheduled today. {data?.nextPatient} is up next!
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => navigate("/")}
                className="flex items-center gap-2 px-4 py-2 bg-[#ff9800] text-white rounded-full font-bold text-sm hover:opacity-90 transition-all"
              >
                <span className="material-symbols-outlined text-sm">child_care</span>
                Switch to Child View
              </button>
              {/* from GET /api/therapist/dashboard → .dailyGoalsMet, .dailyGoalsTarget */}
              <div className="bg-white/70 backdrop-blur px-4 py-2 rounded-xl flex items-center gap-3">
                <div className="w-9 h-9 bg-[#86f898] rounded-full flex items-center justify-center text-[#006a2b]">
                  <span className="material-symbols-outlined text-sm">check_circle</span>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider font-bold text-[#4e5c71]">Daily Progress</p>
                  <p className="text-sm font-extrabold text-[#212f42]">{data?.dailyGoalsMet}/{data?.dailyGoalsTarget} Goals Met</p>
                </div>
              </div>
            </div>
          </section>

          {/* Bento Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">

            {/* Patient list */}
            <div className="lg:col-span-4 space-y-3">
              <h3 className="font-bold text-[#212f42] text-sm uppercase tracking-widest">Your Patients</h3>
              {(data?.patients ?? []).map(p => (
                <button
                  key={p.id}
                  onClick={() => loadPatient(p.id)}
                  className="w-full bg-white rounded-2xl p-5 text-left shadow-sm hover:shadow-md transition-all flex items-center gap-4 border-2 hover:border-[#874e00]/20"
                  style={{ borderColor: patient?.id === p.id ? "#874e00" : "transparent" }}
                >
                  <div className="w-12 h-12 rounded-full bg-[#ff9800] flex items-center justify-center text-white font-black text-lg shrink-0">
                    {p.name[0]}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-[#212f42] text-sm">{p.name}</span>
                      <span className="px-2 py-0.5 bg-[#86f898] text-[#00481b] text-[9px] font-bold rounded-full uppercase">Active</span>
                    </div>
                    <p className="text-xs text-[#4e5c71] truncate">{p.focusArea}</p>
                    <p className="text-xs font-bold text-[#006a2b] mt-1">Avg: {p.avgAccuracy}%</p>
                  </div>
                </button>
              ))}
            </div>

            {/* Patient detail */}
            {patient && (
              <div className="lg:col-span-8 space-y-5">

                {/* Patient header */}
                {/* from GET /api/therapist/patient/:id */}
                <div className="bg-white rounded-2xl p-7 shadow-sm relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-6 opacity-5">
                    <span className="material-symbols-outlined text-9xl">child_care</span>
                  </div>
                  <div className="flex items-center gap-5 relative z-10">
                    <div className="w-20 h-20 rounded-full border-4 border-[#ff9800] bg-gradient-to-br from-[#ff9800] to-[#874e00] flex items-center justify-center text-white text-3xl font-black shadow-lg">
                      {patient.name[0]}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 flex-wrap">
                        <h3 className="text-xl font-extrabold text-[#212f42]">Patient: {patient.name}</h3>
                        <span className="px-3 py-1 bg-[#86f898] text-[#00481b] text-[10px] font-bold rounded-full uppercase tracking-widest">Active</span>
                      </div>
                      <p className="text-[#4e5c71] text-sm mt-1">{patient.focusArea}</p>
                      <div className="flex gap-5 mt-3">
                        <div>
                          <p className="text-xs text-[#4e5c71]">Age</p>
                          <p className="font-bold text-sm">{patient.age} yrs</p>
                        </div>
                        <div className="border-l border-[#a0aec5]/30 pl-5">
                          <p className="text-xs text-[#4e5c71]">Sessions</p>
                          <p className="font-bold text-sm">{patient.sessionsCompleted}/{patient.totalSessions}</p>
                        </div>
                        <div className="border-l border-[#a0aec5]/30 pl-5">
                          <p className="text-xs text-[#4e5c71]">Avg Accuracy</p>
                          <p className="font-bold text-sm text-[#006a2b]">{patient.avgAccuracy}%</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Weekly trend chart */}
                {/* from GET /api/therapist/patient/:id → .weeklyTrend[] */}
                <div className="bg-[#eaf1ff] rounded-2xl p-6">
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h4 className="font-bold text-[#212f42]">Weekly Accuracy Trend</h4>
                      <p className="text-sm text-[#4e5c71]">Progress across last 7 sessions</p>
                    </div>
                    <div className="flex items-center gap-4 text-xs">
                      <span className="flex items-center gap-1">
                        <span className="w-3 h-3 rounded-full bg-[#00618f] inline-block" />
                        Actual
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="w-3 h-3 rounded-full border-2 border-dashed border-[#4e5c71] inline-block" />
                        Target (80%)
                      </span>
                    </div>
                  </div>
                  <div className="h-40 flex items-end justify-between px-2 relative gap-2">
                    {/* 80% target line (at 80% of height = 32px from top in 40px container) */}
                    <div className="absolute left-0 right-0 border-t-2 border-dashed border-[#a0aec5]/40" style={{ bottom: "80%" }} />
                    {(patient.weeklyTrend ?? []).map((d, i) => (
                      <div key={i} className="flex flex-col items-center gap-1 flex-1 group cursor-pointer">
                        <div
                          className="w-full rounded-t-xl relative"
                          style={{
                            height: `${d.accuracy}%`,
                            background: i === patient.weeklyTrend.length - 1 ? "#00618f" : "#4bbaff40",
                            transition: "all 0.5s ease",
                          }}
                        >
                          <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-3 h-3 bg-[#00618f] rounded-full shadow ring-4 ring-[#eaf1ff]" />
                        </div>
                        <span className="text-[9px] font-bold text-[#4e5c71]">{d.accuracy}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Growth metrics */}
                <div className="bg-white rounded-2xl p-6 shadow-sm">
                  <h4 className="text-xs font-extrabold text-[#212f42] mb-5 uppercase tracking-widest">Growth Metrics</h4>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { label: "Exercises", value: `${patient.exercisesCompleted}%`, pct: patient.exercisesCompleted, color: "#006a2b" },
                      { label: "Level", value: `${patient.currentLevel}/${patient.maxLevel}`, pct: (patient.currentLevel / patient.maxLevel) * 100, color: "#874e00" },
                      { label: "Perfect", value: `${patient.perfectScores}/5`, pct: (patient.perfectScores / 5) * 100, color: "#00618f" },
                    ].map(metric => (
                      <div key={metric.label} className="flex flex-col items-center gap-2">
                        <div className="relative w-16 h-16">
                          <svg className="w-full h-full -rotate-90">
                            <circle className="text-[#dce9ff]" cx="32" cy="32" fill="transparent" r="28" stroke="currentColor" strokeWidth="6" />
                            <circle
                              cx="32" cy="32" fill="transparent" r="28"
                              stroke={metric.color}
                              strokeDasharray="176"
                              strokeDashoffset={176 - (metric.pct / 100) * 176}
                              strokeWidth="6"
                            />
                          </svg>
                          <div className="absolute inset-0 flex items-center justify-center text-[9px] font-black text-[#212f42]">
                            {metric.value}
                          </div>
                        </div>
                        <p className="text-[10px] font-bold text-[#4e5c71] text-center">{metric.label}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recent sessions table */}
                {/* from GET /api/therapist/patient/:id → .recentSessions[] */}
                <div className="bg-white rounded-2xl p-6 shadow-sm">
                  <div className="flex justify-between items-center mb-5">
                    <h4 className="font-bold text-[#212f42]">Recent Sessions</h4>
                    <button className="text-[#874e00] font-bold text-sm hover:underline">View All</button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left">
                      <thead>
                        <tr className="text-[10px] font-bold text-[#4e5c71] uppercase tracking-widest border-b border-[#a0aec5]/10">
                          <th className="pb-3">Date</th>
                          <th className="pb-3">Exercise</th>
                          <th className="pb-3">Accuracy</th>
                          <th className="pb-3">Status</th>
                          <th className="pb-3 text-right">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="text-sm">
                        {(patient.recentSessions ?? []).map((s, i) => (
                          <tr key={i} className="hover:bg-[#eaf1ff] transition-colors">
                            <td className="py-3 text-[#4e5c71] text-xs">{s.date}</td>
                            <td className="py-3 font-bold text-[#212f42]">{s.exercise}</td>
                            <td className="py-3 font-extrabold" style={{ color: s.accuracy >= 80 ? "#006a2b" : "#874e00" }}>
                              {s.accuracy}%
                            </td>
                            <td className="py-3">
                              <span
                                className="px-2 py-1 rounded-full text-[10px] font-bold"
                                style={{
                                  background: s.status === "target-met" ? "#86f898" : "#ff980020",
                                  color: s.status === "target-met" ? "#00481b" : "#874e00",
                                }}
                              >
                                {s.status === "target-met" ? "Target Met" : "Near Target"}
                              </span>
                            </td>
                            <td className="py-3 text-right">
                              {/* API PLACEHOLDER: View session detail / recording */}
                              <button className="p-1 hover:bg-white rounded-full text-[#874e00]">
                                <span className="material-symbols-outlined text-sm">visibility</span>
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

              </div>
            )}
          </div>
        </main>
      </div>

      {/* Mobile bottom nav */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-white shadow-lg px-5 py-3 flex justify-between items-center z-50">
        {[
          { icon: "dashboard", label: "Home" },
          { icon: "group", label: "Patients" },
          { icon: "add", label: "", fab: true },
          { icon: "assessment", label: "Reports" },
          { icon: "settings", label: "Admin" },
        ].map(item => item.fab ? (
          <div key="fab" className="relative -top-6">
            <button className="w-14 h-14 bg-[#ff9800] text-white rounded-full shadow-xl flex items-center justify-center">
              <span className="material-symbols-outlined text-2xl">add</span>
            </button>
          </div>
        ) : (
          <button key={item.label} className="flex flex-col items-center gap-1 text-[#4e5c71]">
            <span className="material-symbols-outlined text-sm">{item.icon}</span>
            <span className="text-[9px] font-bold uppercase">{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
