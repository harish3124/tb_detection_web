import "./scss/App.scss";
import { useState } from "react";
import NavButton from "./components/NavButton";
import { Home, About, Models, Predict } from "./views";

import { ReactComponent as VirusSvg } from "./assets/virus.svg";
import VirusBG from "./assets/virus-background.png";

function App() {
  const [view, setView] = useState("home");

  const viewMap = {
    home: <Home viewSetter={setView} />,
    about: <About />,
    models: <Models />,
    predict: <Predict />,
  };

  return (
    <div className="App">
      <img src={VirusBG} alt="Virus Background" className="img-bg" id="img-1" />
      <img src={VirusBG} alt="Virus Background" className="img-bg" id="img-2" />
      <div id="title">
        <VirusSvg style={{ height: "2rem" }} />
        <div>Tuberculosis</div>
      </div>
      <div id="nav">
        <div>
          <NavButton name="Home" viewSetter={setView} />
          <NavButton name="About" viewSetter={setView} />
          <NavButton name="Models" viewSetter={setView} />
          <NavButton name="Predict" viewSetter={setView} />
        </div>
      </div>
      <div id="content">{viewMap[view]}</div>
    </div>
  );
}

export default App;
