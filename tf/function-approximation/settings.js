function f(x) {
  return x.pow(2)
}

function init_model() {

  model.add(tf.layers.dense({
    units: 10,
    inputShape: [1],
    kernelInitializer: 'randomNormal',
    biasInitializer: 'randomNormal'
  }));

  model.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'randomNormal',
    biasInitializer: 'randomNormal'
  }));
  model.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'randomNormal',
    biasInitializer: 'randomNormal'
  }));
  model.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'randomNormal',
    biasInitializer: 'randomNormal'
  }));
  model.add(tf.layers.dense({
    units: 1,
    kernelInitializer: 'randomNormal',
    biasInitializer: 'randomNormal'
  }));

  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(0.1)
  });
}

function min_x() {
  return -1
}

function max_x() {
  return 1
}

function x_space() {
  return tf.linspace(min_x(), max_x(), 101)
}
