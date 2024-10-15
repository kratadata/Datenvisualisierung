// poseNet
let video;
let poseNet;
let poses = [];
var poseNetOptions = {
  detectionType: "single",
};
let tracking_points = ["nose", "leftEye", "rightEye"]

// neuralNetwork
let brain;
let positionSlider;

// interface
let samples = 0;
let positionLerped = 0;
let showPoints = true;
let circleSize = 60;
let trails = [];
const MAX_POS = 10;


function setup() {
  cvs = createCanvas(640, 480);
  cvs.parent("videoContainer");
  fill(255);
  noStroke();
  textAlign(CENTER, CENTER);

  //Video
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  // poseNet
  poseNet = ml5.poseNet(video, poseNetOptions, modelReady);
  poseNet.on("pose", function (results) {
    poses = results;
  });
 
  setButtons();

  // Create the model with following options
  let options = {
    task: "regression",
    inputs: 3,
    outputs: 1,
    debug: true,
  };

  brain = ml5.neuralNetwork(options);

}
function setButtons() { 
  select("#AddPose").mousePressed(() => addPose());
  select("#Train").mousePressed(() => trainModel());
  select("#Predict").mousePressed(() => predict());
  select("#Save").mousePressed(() =>  brain.save());
  const loadButton = select("#Load");
  loadButton.changed(() => {
    brain.load(loadButton.elt.files);
    console.log("Custom Model Loaded!");
  });

  positionSlider = createSlider(0, 200, 0).position(10, 10);
}

// Callback function when poseNet is ready
function modelReady() {
  console.log("model loaded");
}

// Add training examples
function addPose() {
  if (poses.length > 0) {
    inputs = getInputs();
    brain.addData(inputs, [positionSlider.value()]);
    console.log(inputs);
  }

  let amountName = select("#AmountPoses");
  let samples = parseInt(amountName.elt.innerHTML);
  amountName.elt.innerHTML = ++samples;
}

function getInputs() {
  let inputs = [];
  let keypoints = poses[0].pose.keypoints;
  for (let i = 0; i < keypoints.length; i++) {
    if (tracking_points.includes(keypoints[i].part)) {
      inputs.push(keypoints[i].position.x);
      inputs.push(keypoints[i].position.y);
    } 
  }
  return inputs;
} 

// Normalize model
function normalizeModel() {
  brain.normalizeData();
  console.log("data normalized");
}

// Train model
function trainModel() {
  normalizeModel();
  const trainingOptions = {
    epochs: 8,
    batchSize: 12,
  };
  brain.train(trainingOptions, finishedTraining);
}

// Callback function when training is over
function finishedTraining() {
  console.log("model trained");
  showPoints = false;
}

// Make a prediction
function predict() {
  console.log("predicting");
  if (poses.length > 0) {
    let inputs = getInputs();
    console.log(inputs)
    brain.predict(inputs, gotResults);
  }

  if (!inputs || inputs.length === 0) {
    console.error("Invalid inputs: ", inputs);
  } 

} 

// Callback function when new results(poses) are found
function gotResults(error, results) {
  let lerpFactor = 0.05;
  positionLerped = lerp(positionLerped, results[0].value, lerpFactor);
  positionSlider.value(positionLerped);
  predict();
}

function draw() {
  clear();
  textSize(16);
  fill(0);

  // Draw video and mirror it
  push();
  scale(-1, 1);
  translate(-width, 0);
  image(video, 0, 0, width, height);
  //unmirror text
  push()
  translate(width, height-50);
  scale(-1, 1);
  text("Press 'p' to show/hide points", 120, 20);
  pop()
  
  // Draw keypoints
  if (showPoints){
    if (poses.length > 0) {
      let pose = poses[0].pose;
      for (let i = 0; i < pose.keypoints.length; i++) {
        ellipse(pose.keypoints[i].position.x, pose.keypoints[i].position.y, 8);
      }
    }
  }
  pop();
 
  const posX = map(positionSlider.value(), 0, 200, circleSize / 2, width - circleSize / 2);
  const maxCircle = circleSize / MAX_POS;

  // Draw trail
  trails.push({ x: posX, y: height / 2 });

  // Remove poses that are older than MAX_POS
  if (trails.length > MAX_POS) {
    trails.shift();
  }

  for (let i = 0; i < trails.length; i++) {
   circle(trails[i].x, trails[i].y, i * maxCircle);
  }
}

function keyPressed() {
  if (key === "p") {
    showPoints = !showPoints;
  }
}