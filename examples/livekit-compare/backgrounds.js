// Virtual backgrounds drawn on a canvas at load time: no image files, nothing to license.
// Each returns a JPEG blob URL (both processors take an image URL).

const W = 1920
const H = 1080

function canvas(draw) {
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  draw(c.getContext('2d'))
  return new Promise((r) => c.toBlob((b) => r(URL.createObjectURL(b)), 'image/jpeg', 0.92))
}

// Seeded random, so the backgrounds look the same on every load.
function rng(seed) {
  return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646
}

// Warm studio wall: graded backdrop, soft key light, vignette.
const studio = () => canvas((g) => {
  const base = g.createLinearGradient(0, 0, 0, H)
  base.addColorStop(0, '#3a2c28')
  base.addColorStop(1, '#1a1311')
  g.fillStyle = base
  g.fillRect(0, 0, W, H)
  const key = g.createRadialGradient(W * 0.62, H * 0.38, 0, W * 0.62, H * 0.38, W * 0.55)
  key.addColorStop(0, 'rgba(255,196,150,0.55)')
  key.addColorStop(1, 'rgba(255,196,150,0)')
  g.fillStyle = key
  g.fillRect(0, 0, W, H)
  const vig = g.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, W * 0.75)
  vig.addColorStop(0, 'rgba(0,0,0,0)')
  vig.addColorStop(1, 'rgba(0,0,0,0.6)')
  g.fillStyle = vig
  g.fillRect(0, 0, W, H)
})

// Evening city bokeh: blurred light discs over a deep blue gradient.
const bokeh = () => canvas((g) => {
  const base = g.createLinearGradient(0, 0, W, H)
  base.addColorStop(0, '#0b1430')
  base.addColorStop(1, '#1d2f5c')
  g.fillStyle = base
  g.fillRect(0, 0, W, H)
  const r = rng(7)
  const hues = [38, 45, 200, 320, 28]
  g.filter = 'blur(18px)'
  for (let i = 0; i < 70; i++) {
    const rad = 30 + r() * 90
    g.fillStyle = `hsla(${hues[i % hues.length]}, 90%, ${55 + r() * 20}%, ${0.18 + r() * 0.35})`
    g.beginPath()
    g.arc(r() * W, H * 0.15 + r() * H * 0.7, rad, 0, Math.PI * 2)
    g.fill()
  }
  g.filter = 'none'
})

// Mountain dusk: gradient sky, sun, layered ridges.
const dusk = () => canvas((g) => {
  const sky = g.createLinearGradient(0, 0, 0, H)
  sky.addColorStop(0, '#2b2d5e')
  sky.addColorStop(0.55, '#d9747a')
  sky.addColorStop(0.8, '#f6c28b')
  g.fillStyle = sky
  g.fillRect(0, 0, W, H)
  const sun = g.createRadialGradient(W * 0.3, H * 0.62, 0, W * 0.3, H * 0.62, 260)
  sun.addColorStop(0, 'rgba(255,240,200,0.95)')
  sun.addColorStop(0.25, 'rgba(255,220,170,0.6)')
  sun.addColorStop(1, 'rgba(255,220,170,0)')
  g.fillStyle = sun
  g.fillRect(0, 0, W, H)
  const r = rng(3)
  const layers = ['#6b4a6e', '#4a3557', '#2f2340', '#1b1528']
  layers.forEach((color, i) => {
    const y0 = H * (0.62 + i * 0.09)
    g.fillStyle = color
    g.beginPath()
    g.moveTo(0, H)
    let y = y0
    for (let x = 0; x <= W; x += 40) {
      y = Math.min(H, Math.max(H * 0.45, y + (r() - 0.5) * (70 - i * 10)))
      g.lineTo(x, y)
    }
    g.lineTo(W, H)
    g.fill()
  })
})

export async function makeBackgrounds() {
  const [a, b, c] = await Promise.all([studio(), bokeh(), dusk()])
  return [
    { name: 'Studio', url: a },
    { name: 'City lights', url: b },
    { name: 'Dusk', url: c },
  ]
}
