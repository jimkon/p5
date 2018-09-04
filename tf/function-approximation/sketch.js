const model = tf.sequential();;
const convergence = 0.000001

let x_range;

//tensors
let xs, ys;
//ploting values
let x_axis, y_axis, y_pred;
//will be calculated in the process
let y_range;

let losses = []

let pause_flag = false;


function calculate_ys(x) {
  return f(xs)
}

function setup() {
  createCanvas(800, 600)
  frameRate(10)

  init_model()

  x_range = [min_x(), max_x()]

  xs = x_space()
  ys = calculate_ys()

  x_axis = xs.dataSync()
  y_axis = Array.from(ys.dataSync())


  y_range = [min(y_axis), max(y_axis)]
    // console.log(min(y_axis[0], y_axis[1]))
}

function draw() {

  model.fit(xs, ys).then(h => {
    console.log("Loss after : " + h.history.loss[0] + " memory leak: " + tf
      .memory().numTensors);
    losses.push(h.history.loss[0]);
    if (h.history.loss[0] < convergence) {
      console.log(build_graph(model))
      console.log('Convergered to loss', convergence)
        // console.log('Loss history', losses)
      noLoop();
    }
  });

  y_pred = model.predict(xs.expandDims(1)).dataSync()


  background(0)
  strokeWeight(1)
  noFill()

  //plot function and function approximation
  stroke(200)
  rect(0, 0, width / 2, height / 2)
  line(0, height / 4, width / 2, height / 4)
  line(width / 4, 0, width / 4, height / 2)

  //f(x)
  stroke(0, 255, 0, 127)
  strokeWeight(1)
  text("f(x)", 10, 10)
  strokeWeight(5)
  for (var i = 0; i < x_axis.length - 1; i++) {
    let x1 = map(x_axis[i], x_range[0], x_range[1], 0, width / 2);
    let y1 = map(y_axis[i], y_range[0], y_range[1], height / 2, 0);
    let x2 = map(x_axis[i + 1], x_range[0], x_range[1], 0, width / 2);
    let y2 = map(y_axis[i + 1], y_range[0], y_range[1], height / 2, 0);

    line(x1, y1, x2, y2)
  }

  //approximation of f(x)
  stroke(255, 0, 0, 127)
  strokeWeight(1)
  text("f_th(x)", 10, 30)
  strokeWeight(2)
  for (var i = 0; i < x_axis.length - 1; i++) {
    let x1 = map(x_axis[i], x_range[0], x_range[1], 0, width / 2);
    let y1 = map(y_pred[i], y_range[0], y_range[1], height / 2, 0);
    let x2 = map(x_axis[i + 1], x_range[0], x_range[1], 0, width / 2);
    let y2 = map(y_pred[i + 1], y_range[0], y_range[1], height / 2, 0);

    line(x1, y1, x2, y2)
  }

  //loss history
  stroke(200)
  line(width / 2, 0.9 * height / 2, width, 0.9 * height / 2)

  stroke(0, 255, 0)
  strokeWeight(1)
  text("loss", width / 2 + 10, 10)
  strokeWeight(2)

  let x_loss = losses.length
  let max_loss = max(losses)
  for (var i = 0; i < losses.length - 1; i++) {
    let x1 = map(i, 0, x_loss, 1.1 * width / 2, width)
    let y1 = map(losses[i], 0, max_loss, 0.9 * height / 2, 0)
    let x2 = map(i + 1, 0, x_loss, 1.1 * width / 2, width)
    let y2 = map(losses[i + 1], 0, max_loss, 0.9 * height / 2, 0)
    if (i % 9 == 0) {
      stroke(200)
      strokeWeight(1)
      line(x1, 0, x1, height / 2)
      stroke(0, 255, 0)
      strokeWeight(2)
    }
    line(x1, y1, x2, y2)
  }

  //model graph
  show(model, 0, height / 2, width, height / 2)

}

function keyPressed() {
  if (!pause_flag) {
    noLoop()
    console.log(build_graph(model))
  } else {
    loop()
  }
  pause_flag = !pause_flag

}
