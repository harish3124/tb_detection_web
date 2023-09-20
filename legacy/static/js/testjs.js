const inputAudio = document.getElementById("input-audio");
const uploadButton = document.getElementById("upload-button");
const audioElement = document.getElementById("audio-element");
const serverURL = "http://localhost:5000/upload";

inputAudio.addEventListener("change", (event) => {
  const file = event.target.files[0];
  audioElement.src = URL.createObjectURL(file);
});

uploadButton.addEventListener("click", () => {
  const xhr = new XMLHttpRequest();
  xhr.open("POST", serverURL, true);
  const formData = new FormData();
  formData.append("file", inputAudio.files[0]);
  xhr.send(formData);
});
