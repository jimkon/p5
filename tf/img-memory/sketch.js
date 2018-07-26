let img, ev_img;
const img_size = 10;
const convergence = 0.000001
const model = tf.sequential();


let x_tensor, y_tensor;

function img_f(x1, x2) {

  return (x1 + x2) / 2
}

function init_model() {
  model.add(tf.layers.dense({
    units: 3,
    inputShape: [2],
    kernelInitializer: 'randomNormal',
    biasInitializer: 'randomNormal'
      // useBias: false
  }));
  model.add(tf.layers.dense({
    units: 1,
    kernelInitializer: 'randomNormal',
    biasInitializer: 'randomNormal'
  }));
  // model.add(tf.layers.dense({
  //   inputShape: [2],
  //   units: 1,
  //   kernelInitializer: 'randomNormal',
  //   biasInitializer: 'randomNormal'
  // }));


  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.sgd(0.5)
  });

}

function setup() {
  createCanvas(600, 600)
  frameRate(10)
  img = createImage(img_size, img_size)
  ev_img = createImage(img_size, img_size)
  for (var i = 0; i < img_size; i++) {
    for (var j = 0; j < img_size; j++) {
      img.set(i, j, map(img_f(i / img_size, j / img_size), 0, 1, 0, 255))
    }
  }
  img.updatePixels()

  init_model()

  let xs = []
  let ys = []
  for (var i = 0; i < img_size; i++) {
    for (var j = 0; j < img_size; j++) {
      let x1 = i / (img_size - 1)
      let x2 = j / (img_size - 1)
      xs = append(xs, [x1, x2])
      ys = append(ys, img_f(x1, x2))
    }
  }
  x_tensor = tf.tensor2d(xs)
  y_tensor = tf.tensor1d(ys)

  // evaluate()
}

function draw() {
  model.fit(x_tensor, y_tensor).then(h => {
    console.log("Loss after : " + h.history.loss[0] + " memory leak: " + tf
      .memory().numTensors);
    if (h.history.loss[0] < convergence) {
      console.log(build_graph(model))
      console.log('Convergered to loss', convergence)
      noLoop();
    } else {
      evaluate()
    }
  });


  // noLoop()
  background(0, 255, 0)
  image(img, width / 4 - 10 * img_size, height / 5 - 10 * img_size, 20 *
    img_size, 20 *
    img_size)
  image(ev_img, 3 * width / 4 - 10 * img_size, height / 5 - 10 *
    img_size, 20 *
    img_size,
    20 *
    img_size)
  show(model, 0, height / 2, width, height / 2)

}

function evaluate() {
  let xs = x_tensor.dataSync()
    // let ys = y_tensor.dataSync()
  let ys = model.predict(x_tensor).dataSync()
    // console.log(ys)
    // noLoop()
  for (var i = 0; i < ys.length; i++) {
    let x1 = xs[2 * i]
    let x2 = xs[2 * i + 1]
    let y = ys[i]
      // let y = img_f(x1, x2)
    ev_img.set(x1 * (img_size - 1), x2 * (img_size - 1), map(y, 0, 1, 0, 255))
  }

  ev_img.updatePixels()
}

function keyPressed() {
  noLoop();
}
