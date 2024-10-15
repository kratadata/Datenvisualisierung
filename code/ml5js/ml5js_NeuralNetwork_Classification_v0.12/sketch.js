let faceMesh;
let video;
let faces = [];
let options = { maxFaces: 1, refineLandmarks: false, flipHorizontal: false };
let label = "";
let brain;
let showPoints = true;


function setup() {
  createCanvas(640, 480);
 
  // Create the webcam video and hide it
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();
  setButtons();

  // Create the facemesh model
  faceMesh = ml5.facemesh(video, modelReady);  
  faceMesh.on("face", results => {
    faces = results;
  });

  // Create the model with options
  let options = {
    task: "classification",
    debug: true
  };

  brain = ml5.neuralNetwork(options);

}

function modelReady() {
  console.log("model loaded");
}

function draw() {
  // Draw the webcam video
  image(video, 0, 0, width, height);
  textSize(16);
  fill(255);
  text("Press 'p' to show/hide points", 120, 20);
    
  textAlign(CENTER, CENTER);
  textSize(64);
  text(label, width / 2, height / 2);

  if (showPoints){
    // Draw all the tracked face points
    for (let i = 0; i < faces.length; i++) {
      let keypoints = faces[i].scaledMesh;
      for (let j = 0; j < keypoints.length; j++) {
        let keypoint = keypoints[j];
        noStroke();
        circle(keypoint[0], keypoint[1], 5);
      }
    }
  } 
}

function getInputs() {
  let inputs = [];
  for (let i = 0; i < faces.length; i++) {
    let keypoints = faces[i].scaledMesh;
    for (let j = 0; j < keypoints.length; j++) {
      let keypoint = keypoints[j];
      inputs.push(keypoint[0]);
      inputs.push(keypoint[1]);
    }
  }
  return inputs;
}


function setButtons() {
  select("#AddKeypoints").mousePressed(() => addExample());
  select("#Train").mousePressed(() => trainModel());
  select("#Classify").mousePressed(() => classify());
  select("#Save").mousePressed(() => brain.save());
  const loadButton = select("#Load");
  loadButton.changed(() => {
    brain.load(loadButton.elt.files);
    console.log("Custom Model Loaded!");
  });
}

// Add training examples
function addExample() {
  if (faces.length > 0) {
    inputs = getInputs();
  }

  brain.addData(inputs, [select("#Input").value()]);
  console.log(inputs);
  let amountName = select("#AmountKeypoints");
  let samples = parseInt(amountName.elt.innerHTML);
  amountName.elt.innerHTML = ++samples;
}

// Normalize model
function normalizeModel() {
  brain.normalizeData();
}

// Train model
function trainModel() {
  normalizeModel();
  const trainingOptions = {
    epochs: 32,
    batchSize: 12,
  };
  brain.train(trainingOptions, finishedTraining);
}

function finishedTraining() {
  classify();
  showPoints = false;
}

function classify() {
  console.log("Classifying");
  let inputs = getInputs();
 
  if (!inputs || inputs.length === 0) {
    console.error("Invalid inputs: ", inputs);
    // Try classifying again
    //setTimeout(classify, 1000);
  } 
  brain.classify(inputs, handleResults);
}

function handleResults(error, results) {
  if (error) {
    console.error(error);
    return;
  }
  label = results[0].label;
  classify();
}

function keyPressed() {
  if (key === "p") {
    showPoints = !showPoints;
  }
}