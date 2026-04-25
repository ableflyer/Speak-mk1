import { useState } from "react";
import { useTheme } from "../context/ThemeContext";
import { THEMES, THEME_ORDER } from "../themes/themes";

export default function ThemePicker({ onClose }) {
  const { themeId, changeTheme } = useTheme();
  const [selected, setSelected] = useState(themeId);

  function handleApply() {
    changeTheme(selected);
    onClose();
  }

  return (
    // Backdrop
    <div
      className="fixed inset-0 z-[100] flex items-end sm:items-center justify-center"
      style={{ background: "rgba(0,0,0,0.4)", backdropFilter: "blur(6px)" }}
      onClick={onClose}
    >
      {/* Sheet */}
      <div
        className="w-full sm:max-w-md bg-white rounded-t-3xl sm:rounded-3xl p-6 shadow-2xl"
        onClick={e => e.stopPropagation()}
        style={{ animation: "slideUp 0.3s cubic-bezier(.34,1.56,.64,1)" }}
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="font-headline font-extrabold text-xl text-gray-900">Choose Your Theme</h2>
            <p className="text-sm text-gray-500 mt-0.5">Pick how your dashboard looks</p>
          </div>
          <button
            onClick={onClose}
            className="w-9 h-9 rounded-full bg-gray-100 flex items-center justify-center text-gray-500 hover:bg-gray-200 transition-all"
          >
            <span className="material-symbols-outlined text-sm">close</span>
          </button>
        </div>

        {/* Theme cards */}
        <div className="grid grid-cols-2 gap-3 mb-6">
          {THEME_ORDER.map(id => {
            const t = THEMES[id];
            const isSelected = selected === id;
            return (
              <button
                key={id}
                onClick={() => setSelected(id)}
                className="relative rounded-2xl p-4 text-left transition-all active:scale-95"
                style={{
                  background: t.preview.bg,
                  border: isSelected ? `3px solid ${t.preview.accent}` : "3px solid transparent",
                  boxShadow: isSelected ? `0 0 0 2px ${t.preview.accent}30` : "0 2px 8px rgba(0,0,0,0.06)",
                }}
              >
                {/* Check mark */}
                {isSelected && (
                  <div
                    className="absolute top-2 right-2 w-6 h-6 rounded-full flex items-center justify-center"
                    style={{ background: t.preview.accent }}
                  >
                    <span className="material-symbols-outlined text-white text-sm" style={{ fontVariationSettings: "'FILL' 1", fontSize: 14 }}>check</span>
                  </div>
                )}

                {/* Preview swatch */}
                <div className="flex gap-1 mb-3">
                  <div className="w-5 h-5 rounded-full shadow-sm" style={{ background: t.preview.accent }} />
                  <div className="w-5 h-5 rounded-full shadow-sm" style={{ background: t.preview.secondary }} />
                  <div className="w-5 h-5 rounded-full shadow-sm border border-gray-200" style={{ background: t.preview.bg }} />
                </div>

                {/* Mini dashboard preview */}
                <div className="rounded-xl overflow-hidden mb-3" style={{ background: t.preview.bg, border: `1px solid ${t.preview.accent}20` }}>
                  <div className="h-2 w-full" style={{ background: t.preview.accent, opacity: 0.8 }} />
                  <div className="p-2 space-y-1">
                    <div className="h-1.5 rounded-full w-3/4" style={{ background: t.preview.accent, opacity: 0.4 }} />
                    <div className="h-1.5 rounded-full w-1/2" style={{ background: t.preview.secondary, opacity: 0.3 }} />
                    <div className="flex gap-1 mt-1">
                      {[60, 80, 100, 70, 40].map((h, i) => (
                        <div key={i} className="flex-1 rounded-t" style={{ height: h * 0.12, background: i === 2 ? t.preview.accent : t.preview.secondary, opacity: i === 2 ? 0.9 : 0.3 }} />
                      ))}
                    </div>
                  </div>
                </div>

                <div className="text-2xl mb-1">{t.emoji}</div>
                <div className="font-headline font-bold text-sm" style={{ color: t.preview.accent }}>{t.name}</div>
                <div className="text-xs text-gray-400 mt-0.5">{t.tagline}</div>
              </button>
            );
          })}
        </div>

        {/* Apply button */}
        <button
          onClick={handleApply}
          className="w-full py-4 rounded-2xl font-bold text-white text-base transition-all active:scale-95 shadow-lg"
          style={{ background: THEMES[selected].preview.accent }}
        >
          Apply Theme ✓
        </button>
      </div>

      <style>{`
        @keyframes slideUp {
          from { transform: translateY(40px); opacity: 0; }
          to   { transform: translateY(0);    opacity: 1; }
        }
      `}</style>
    </div>
  );
}
