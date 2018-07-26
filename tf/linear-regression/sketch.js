

let xs = [];
let ys = [];

let m, b;

const w = 800, h = 600

// const x_range = [0, w];
// const y_range = [0, h];

const x_range = [-1, 1];
const y_range = [-1, 1];


const loss = (predictions, labels)=>predictions.sub(labels).square().mean();
const predict = (x)=>x.mul(m).add(b);
const optimizer = tf.train.sgd(0.5);


function setup() {
  createCanvas(w,h)
  m = tf.variable(tf.scalar(0))
  b = tf.variable(tf.scalar(0))
}



function draw() {


  if (xs.length>0){

    optimizer.minimize(()=>loss(tf.tensor1d(ys), predict(tf.tensor1d(xs))))

  }



  background(0)
  stroke(255)
  for (var i = 0; i < xs.length; i++) {
    let x = map(xs[i], x_range[0], x_range[1], 0, width)
    let y = map(ys[i], y_range[0], y_range[1], height, 0)
    ellipse(x, y, 5, 5)
  }

  stroke(255, 0, 255)

  tf.tidy(()=>{
    let lx = tf.tensor1d(x_range)
    let ly = predict(lx)

    let temp_y = ly.dataSync()
    line(0, map(temp_y[0], y_range[0], y_range[1], height, 0), width, map(temp_y[1], y_range[0], y_range[1], height, 0))

    fill(255)
    m_v = m.dataSync()
    b_v = b.dataSync()
    text("y= "+round(100*m_v)/100+" x + "+round(100*b_v)/100, 10, 10)
  });
  // lx.dispose()
  // ly.dispose()
  // console.log(tf.memory().numTensors)
}

function mousePressed(){
  let x = map(mouseX, 0, width, x_range[0], x_range[1])
  let y = map(mouseY, height, 0, y_range[0], y_range[1])
  xs = append(xs, x)
  ys = append(ys, y)
}
