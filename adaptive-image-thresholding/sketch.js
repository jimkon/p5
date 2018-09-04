let img, processed_img;

function preload() {
  img = loadImage("page.jpg")
  img.loadPixels()
  processed_img = createImage(400, 300)

  processed_img.pixels = arrayCopy(Array.from(img.pixels))
}

function setup() {
  createCanvas(800, 600)
}

function draw() {
  background(0)
  image(img, 0, 0)
  image(processed_img, 400, 0)

}
