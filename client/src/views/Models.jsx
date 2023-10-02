import { useState, useEffect } from "react";

export const Models = () => {
  const [resp, setResp] = useState({});
  useEffect(() => {
    fetch("http://localhost:5000/models", { method: "POST" }).then((res) => {
      res.json().then((res) => {
        setResp(res);
      });
    });
  }, []);
  return (
    <div id="models">
      <div className="text-large"> Accuracy </div>
      <div className="acc">
        Gradient Boosting: <span>{resp.gbc}</span>
      </div>
      <div className="acc">
        Random Forest: <span>{resp.rf}</span>
      </div>
      <div className="acc">
        Convolutional Neural Network: <span>{resp.cnn}</span>
      </div>
      <div className="acc">
        Decision Tree: <span>{resp.dts}</span>
      </div>
    </div>
  );
};
