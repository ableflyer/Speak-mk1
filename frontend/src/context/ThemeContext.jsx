import { createContext, useContext, useState, useEffect } from "react";
import { THEMES, DEFAULT_THEME } from "../themes/themes";

const ThemeContext = createContext(null);

export function ThemeProvider({ children }) {
  const [themeId, setThemeId] = useState(() => {
    // Persist theme choice in localStorage
    return localStorage.getItem("sq_theme") || DEFAULT_THEME;
  });

  const theme = THEMES[themeId] || THEMES[DEFAULT_THEME];

  function changeTheme(id) {
    setThemeId(id);
    localStorage.setItem("sq_theme", id);
  }

  // Inject extra Google Fonts if the theme needs them (e.g. Manrope)
  useEffect(() => {
    if (theme.extraGoogleFont) {
      const id = "theme-font-" + themeId;
      if (!document.getElementById(id)) {
        const link = document.createElement("link");
        link.id = id;
        link.rel = "stylesheet";
        link.href = `https://fonts.googleapis.com/css2?family=${theme.extraGoogleFont}&display=swap`;
        document.head.appendChild(link);
      }
    }
  }, [themeId]);

  return (
    <ThemeContext.Provider value={{ theme, themeId, changeTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);
