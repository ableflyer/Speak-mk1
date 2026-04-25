import { useNavigate, useLocation } from "react-router-dom";

const childNav = [
  { icon: "home_app_logo", label: "Home", path: "/" },
  { icon: "videogame_asset", label: "Games", path: "/adventure" },
  { icon: "workspace_premium", label: "Awards", path: "/" },
  { icon: "explore", label: "Missions", path: "/" },
];

export default function BottomNav({ active, role = "child", colors = {}, fonts = {} }) {
  const navigate = useNavigate();
  const location = useLocation();
  const items = childNav;

  const navActive = colors.navActive || "#006b1b";
  const navInactive = colors.navInactive || "#874e00";

  return (
    <nav
      className="fixed bottom-0 left-0 w-full z-50 flex justify-around items-end px-4 pb-5 pt-2 rounded-t-3xl shadow-lg border-t"
      style={{
        background: "rgba(255,255,255,0.75)",
        backdropFilter: "blur(20px)",
        borderColor: `${navActive}15`,
        fontFamily: fonts.body,
      }}
    >
      {items.map(item => {
        const isActive = item.path === "/" && active === "home"
          ? true
          : location.pathname === item.path && item.path !== "/";
        return (
          <button
            key={item.label}
            onClick={() => navigate(item.path)}
            className="flex flex-col items-center justify-center transition-all duration-300 active:scale-90"
            style={
              isActive
                ? {
                    background: navActive,
                    color: "white",
                    borderRadius: 9999,
                    padding: "8px 16px",
                    transform: "translateY(-6px)",
                    boxShadow: `0 4px 12px ${navActive}40`,
                  }
                : { color: navInactive, padding: "8px 12px" }
            }
          >
            <span
              className="material-symbols-outlined"
              style={{ fontVariationSettings: isActive ? "'FILL' 1" : "'FILL' 0" }}
            >
              {item.icon}
            </span>
            <span className="text-[10px] font-bold mt-0.5">{item.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
