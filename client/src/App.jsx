import "./css/App.css";
import { useState } from "react";
import NavButton from "./components/NavButton";
import { Home, About, Team, Run } from "./views";

const viewMap = {
  home: <Home />,
  about: <About />,
  team: <Team />,
  run: <Run />,
};

function App() {
  const [view, setView] = useState("home");
  return (
    <div className="App">
      <div id="nav">
        <NavButton name="Home" viewSetter={setView} />
        <NavButton name="About" viewSetter={setView} />
        <NavButton name="Team" viewSetter={setView} />
        <NavButton name="Run" viewSetter={setView} />
      </div>
      <div id="content">{viewMap[view]}</div>
    </div>
  );
}

export default App;
