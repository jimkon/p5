function show(model, x, y, w, h) {
  let background_color = 50;
  fill(background_color)
  stroke(background_color)
  strokeWeight(1)
  rect(x, y, w, h)

  let layers = model.layers.length + 1
  let layer_w = w / layers
  let node_size = layer_w * 0.2

  stroke(255)
  fill(255)
    // for (var i = 1; i < layers; i++) {
    //   line(layer_w * i, y, layer_w * i, y + h)
    // }

  let nodes = build_graph(model)

  // console.log(nodes)

  w_range = {
    'min': 1e10,
    'max': -1e10
  }
  b_range = {
    'min': 1e10,
    'max': -1e10
  }

  for (var layer = 0; layer < nodes.length; layer++) {
    let nx = x + layer_w * (layer + 0.5)

    for (var node_index = 0; node_index < nodes[layer].length; node_index++) {
      let ny = y + (node_index + 0.5) * h / nodes[layer].length

      node = nodes[layer][node_index]

      node['x'] = nx
      node['y'] = ny

      if (node['weights']) {
        w_range['min'] = min(min(node['weights']), w_range['min'])
        w_range['max'] = max(max(node['weights']), w_range['max'])
      }

      if (node['bias']) {
        b_range['min'] = min(node['bias'], b_range['min'])
        b_range['max'] = max(node['bias'], b_range['max'])
      }

      // ellipse(nx, ny, node_size)
    }
  }
  // console.log(nodes)
  // console.log(w_range)
  // console.log(b_range)

  let max_range = max([abs(w_range['max']), abs(w_range['min']),
    abs(b_range['max']), abs(b_range['min'])
  ])

  for (var layer = nodes.length - 1; layer >= 0; layer--) {
    for (var node_index = 0; node_index < nodes[layer].length; node_index++) {
      node = nodes[layer][node_index]
        //node coordinates
      let x = node['x']
      let y = node['y']

      let bias = node['bias']

      if (node['weights'] && layer > 0) {
        for (var i = 0; i < node['weights'].length; i++) {
          let weight = node['weights'][i]

          let px = nodes[layer - 1][i]['x']
          let py = nodes[layer - 1][i]['y']


          let color = map(abs(weight), 0, max_range, 0, 255)
          stroke(weight < 0 ? color : 0, weight > 0 ? color : 0, 0)
          strokeWeight(map(abs(weight), 0, max_range, 0.5, 4))
          line(x, y, px, py)
        }
      }

      let color = map(abs(bias), 0, max_range, 0, 255)
      stroke(bias < 0 ? color : 0, bias > 0 ? color : 0, 0)

      fill(background_color)
      strokeWeight(3)
      ellipse(x, y, node_size)
    }
  }


}

function build_graph(model) {

  let nodes = []

  let names = Object.keys(model.getNamedWeights())
    // console.log(names)

  let i = -1;
  let increment = 1;

  while (i < names.length) {
    increment = 1;

    if (i == -1) {
      // special case for input layer. no biases no kernels
      let layer = []
      let size = model.inputLayers[0].batchInputShape[1]
      for (var j = 0; j < size; j++) {
        //node instance
        node = {
          'weights': null,
          'bias': null
        }

        layer.push(node)
          // console.log("node", node)

      }

      nodes.push(layer)
    } else {

      let kernels = model.getNamedWeights()[names[i]].dataSync()

      let biases = null;

      if (i < names.length - 1 &&
        names[i + 1].split("/")[1] == "bias") {

        biases = model.getNamedWeights()[names[i + 1]].dataSync()
        increment = 2

      }

      let kernels_shape = model.getNamedWeights()[names[i]].shape

      // console.log(i, names[i])
      // console.log('kernels', kernels)
      // console.log('biases', biases)

      let size = kernels_shape[1]
      let layer = []

      for (var node_i = 0; node_i < size; node_i++) {
        //node instance
        node = {
          'weights': null,
          'bias': null
        }

        node['weights'] = []

        for (var k = 0; k < kernels_shape[0]; k++) {
          node['weights'].push(kernels[k * size + node_i])
        }

        if (biases) {
          node['bias'] = biases[node_i];
        }

        layer.push(node);
        // console.log("node", node)
      }
      // console.log("nodes", nodes)

      nodes.push(layer);

    }
    i += increment;
  }


  return nodes;
}
