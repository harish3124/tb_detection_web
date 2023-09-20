import React from "react";

const NavButton = ({ name, viewSetter }) => {
  const handleClick = () => {
    viewSetter(name.toLowerCase());
  };
  return (
    <div
      className={`nav-button ${name === "Predict" ? "text-accent" : ""}`}
      onClick={handleClick}
    >
      {name}
    </div>
  );
};

export default NavButton;
