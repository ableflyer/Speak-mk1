import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppProvider } from "./context/AppContext";
import { ThemeProvider } from "./context/ThemeContext";
import ChildDashboard from "./pages/ChildDashboard";
import ExercisePage from "./pages/ExercisePage";
import AdventureTalk from "./pages/AdventureTalk";
import CheckpointPage from "./pages/CheckpointPage";
import TherapistPortal from "./pages/TherapistPortal";

export default function App() {
  return (
    <ThemeProvider>
      <AppProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<ChildDashboard />} />
            <Route path="/exercise" element={<ExercisePage />} />
            <Route path="/adventure" element={<AdventureTalk />} />
            <Route path="/checkpoint" element={<CheckpointPage />} />
            <Route path="/therapist" element={<TherapistPortal />} />
          </Routes>
        </BrowserRouter>
      </AppProvider>
    </ThemeProvider>
  );
}
