let frameIdMap = {};
let frameIdMapStand = {};
let frameIdMapHand = {};

document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  enableHorizontalScrolling();
  setupVideoControl();
});

function setupEventListeners() {
  document
    .getElementById("fileInput")
    .addEventListener("change", handleFileInput);
  document
    .getElementById("folderInput")
    .addEventListener("change", loadVideosFromFolder);
  document
    .getElementById("selectVideo2")
    .addEventListener("change", () =>
      updateFrameDisplay("selectVideo2", "frameVideo2", frameIdMapStand)
    );
  document
    .getElementById("selectVideo3")
    .addEventListener("change", () =>
      updateFrameDisplay("selectVideo3", "frameVideo3", frameIdMapHand)
    );
}

function setupVideoControl() {
  const playPauseButton = document.getElementById("playPauseButton");
  playPauseButton.addEventListener("click", togglePlayPause);
}

function togglePlayPause() {
  const videos = [
    document.getElementById("video1"),
    document.getElementById("video2"),
    document.getElementById("video3"),
  ];

  videos.forEach((video) => {
    if (video.paused) {
      video.play();
    } else {
      video.pause();
    }
  });
}

function loadVideosFromFolder(event) {
  const files = event.target.files;
  const videoFileNames = ["video1.mp4", "video2.mp4", "video3.mp4"];
  const textFileNames = ["Stand.txt", "Hand.txt"];

  for (let file of files) {
    if (videoFileNames.includes(file.name) && file.type.includes("video")) {
      loadVideoFile(file);
    } else if (
      textFileNames.includes(file.name) &&
      file.type.includes("text")
    ) {
      loadTextFile(file, file.name.split(".")[0]);
    }
  }
}

function loadVideoFile(file) {
  const videoElementId = file.name.split(".")[0];
  const videoElement = document.getElementById(videoElementId);
  if (videoElement) {
    videoElement.src = URL.createObjectURL(file);
    videoElement.load();
  }
}

function loadTextFile(file, type) {
  const reader = new FileReader();
  reader.onload = (event) => {
    processTextFileContent(event.target.result, type);
  };
  reader.readAsText(file);
}

function processTextFileContent(content, type) {
  const uniqueIds = new Set();
  content.split("\n").forEach((line) => {
    const [frame, id] = line.trim().split(" ");
    if (frame && id) {
      uniqueIds.add(id);
      if (type === "Stand") {
        frameIdMapStand[id] = frameIdMapStand[id] || [];
        frameIdMapStand[id].push(frame);
      } else if (type === "Hand") {
        frameIdMapHand[id] = frameIdMapHand[id] || [];
        frameIdMapHand[id].push(frame);
      }
    }
  });

  populateSelectElement(
    uniqueIds,
    type === "Stand" ? "selectVideo2" : "selectVideo3"
  );
}

function populateSelectElement(uniqueIds, selectId) {
  const selectElement = document.getElementById(selectId);
  selectElement.innerHTML = "";
  uniqueIds.forEach((id) => {
    selectElement.appendChild(new Option(id, id));
  });
}

function handleFileInput(event) {
  const file = event.target.files[0];
  if (file) {
    document.getElementById("fileName").textContent = file.name;
    readAndProcessFile(file);
  }
}

function readAndProcessFile(file) {
  const reader = new FileReader();
  reader.onload = (event) => processFileContent(event.target.result);
  reader.readAsText(file);
}

function processFileContent(content) {
  frameIdMap = {};
  const uniqueIds = new Set();
  content.split("\n").forEach((line) => {
    const [frame, id] = line.trim().split(" ");
    if (frame && id) {
      uniqueIds.add(id);
      frameIdMap[id] = frameIdMap[id] || [];
      frameIdMap[id].push(frame);
    }
  });
  populateSelectElements(uniqueIds);
}

function populateSelectElements(uniqueIds) {
  const selectElements = [
    document.getElementById("selectVideo2"),
    document.getElementById("selectVideo3"),
  ];
  selectElements.forEach((select) => {
    select.innerHTML = "";
    uniqueIds.forEach((id) => {
      select.appendChild(new Option(id, id));
    });
  });
}

function updateFrameDisplay(selectId, frameDivId, frameMap) {
  const selectedId = document.getElementById(selectId).value;
  const frames = frameMap[selectedId] || [];
  document.getElementById(
    frameDivId
  ).innerHTML = `Frames for id ${selectedId}: ${frames.join(", ")}`;
}

function enableHorizontalScrolling() {
  document.querySelectorAll(".id-frame").forEach((frame) => {
    frame.addEventListener("wheel", (e) => {
      e.preventDefault();
      frame.scrollLeft += e.deltaY;
    });
  });
}
