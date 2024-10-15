let mobilenet;
let video;
let classifier;
let isTrained = false;
let class1Images = [];
let class2Images = [];
let class1Name = "Person";
let class2Name = "Background";
let lossValue;

function setup() {
  cvs = createCanvas(640, 480);
  cvs.parent("videoContainer");
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  featureExtractor = ml5.featureExtractor("MobileNet", modelReady);
  const options = {
    numLabels: 2,
    learningRate: 0.0001,
    hiddenUnits: 100,
    epochs: 20,
    batchSize: 0.4,
    debug: true
  };

  classifier = featureExtractor.classification(video, options, videoReady);
  showButtons();
}

function draw() {
  image(video, 0, 0, width, height);

  if (isTrained) {
    classify(); 
  } 
}

function modelReady() {
  console.log("Model loaded!");
}

function videoReady() {
  console.log("Video is ready!");
}

function addToArray(arrayName, spanName) {
  let img = cvs.elt.toDataURL("image/jpg", 1);
  arrayName.push(img);
  let amountName = select(spanName);
  let amountNumber = parseInt(amountName.elt.innerHTML);
  amountName.elt.innerHTML = ++amountNumber;
}

function showButtons() {
  select("#Class1").mousePressed(() => addToArray(class1Images, "#AmountClass1"));
  select("#Class2").mousePressed(() => addToArray(class2Images, "#AmountClass2"));

  select("#Train").mousePressed(async () => {
    await addTrainingData();
    classifier.train(lossValue => {
      console.log('Loss is', lossValue);
      if (lossValue == null) {
        isTrained = true;
      }
    });
  });

  select("#Save").mousePressed(() => classifier.save());

  select("#Load").changed(() => {
    let files = select("#Load").elt.files;
    classifier.load(files);
  });
}

function addTrainingData() {
  class1Images.forEach(img => classifier.addImage(img, class1Name));
  class2Images.forEach(img => classifier.addImage(img, class2Name));
}

function classify() {
  classifier.classify(video, gotResults);
}

function gotResults(error, results) {
  if (error) {
    console.error(error);
  } else {
    console.log(results);
  }
}