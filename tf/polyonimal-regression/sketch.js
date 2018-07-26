let xs = [];
let ys = [];

let f;

const w = 800,
  h = 600

const x_range = [-1, 1];
const y_range = [-1, 1];

const x_axis = tf.linspace(x_range[0], x_range[1], 100).as2D(100, 1);

const model = tf.sequential();

function setup() {
  createCanvas(w, h)
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
  }));

  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.sgd(0.5)
  });
}



function draw() {


  if (xs.length > 0) {
    // tf.tidy(() => {
    const x = tf.tensor1d(xs).expandDims(1);
    const y = tf.tensor1d(ys).expandDims(1);
    model.fit(x, y).then();
    // model.fit(x, y).then(() => {
    //   x.dispose();
    //   y.dispose();
    // });
    // x.dispose();
    // y.dispose();
    // });



  }



  background(0)
  stroke(255)
  for (var i = 0; i < xs.length; i++) {
    let x = map(xs[i], x_range[0], x_range[1], 0, width)
    let y = map(ys[i], y_range[0], y_range[1], height, 0)
    ellipse(x, y, 5, 5)
  }


  tf.tidy(() => {
    stroke(255, 0, 255)
    let y_axis = model.predict(x_axis)

    let curve_x = x_axis.dataSync()
    let curve_y = y_axis.dataSync()

    for (var i = 0; i < curve_x.length - 1; i++) {

      let px1 = map(curve_x[i], x_range[0], x_range[1], 0, width)
      let py1 = map(curve_y[i], y_range[0], y_range[1], height, 0)
      let px2 = map(curve_x[i + 1], x_range[0], x_range[1], 0, width)
      let py2 = map(curve_y[i + 1], y_range[0], y_range[1], height, 0)

      line(px1, py1, px2, py2)
    }

  });
  // lx.dispose()
  // ly.dispose()
  console.log(tf.memory().numTensors)
}

function mousePressed() {
  let x = map(mouseX, 0, width, x_range[0], x_range[1])
  let y = map(mouseY, height, 0, y_range[0], y_range[1])
  xs = append(xs, x)
  ys = append(ys, y)
}
