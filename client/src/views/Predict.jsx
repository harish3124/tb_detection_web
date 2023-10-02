import { useState } from "react";

export const Predict = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", file);

    // TODO Change Url for Production
    fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    }).then((res) => {
      res.json().then((res) => {
        setResult(res.result);
      });
    });
  };
  return (
    <div id="predict">
      {result !== null && (
        <div>
          You <span>{result ? "" : "DONT"} HAVE </span> Tuberculosis
        </div>
      )}
      <form onSubmit={handleSubmit}>
        <div>
          <input
            type="file"
            accept="audio/*"
            name="file"
            required={true}
            onChange={(e) => setFile(e.target.files[0])}
          />
        </div>
        <input type="submit" />
      </form>
    </div>
  );
};
