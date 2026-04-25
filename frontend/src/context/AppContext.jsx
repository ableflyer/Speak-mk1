import { createContext, useContext, useState } from "react";

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [user, setUser] = useState({
    id: "user-001",
    name: "Sarah",
    role: "child", // "child" | "therapist"
  });
  const [sessionId, setSessionId] = useState("sess-001");
  const [currentExerciseIndex, setCurrentExerciseIndex] = useState(0);
  const [exercises, setExercises] = useState([]);
  const [sessionResults, setSessionResults] = useState([]);

  return (
    <AppContext.Provider value={{
      user, setUser,
      sessionId, setSessionId,
      currentExerciseIndex, setCurrentExerciseIndex,
      exercises, setExercises,
      sessionResults, setSessionResults,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export const useApp = () => useContext(AppContext);
