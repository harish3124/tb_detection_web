export const Home = ({ viewSetter }) => {
  return (
    <div className="outline-full">
      <div className="text-large">
        Deep Learning Based <span>TUBERCULOSIS</span> Detection Using Cough
        Analysis
      </div>
      <div id="btn-predict" onClick={() => viewSetter("predict")}>
        Predict
      </div>
    </div>
  );
};
