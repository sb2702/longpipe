// node tools/smoke.mjs [--nowebgpu] — headless Chrome smoke test of model: 'mediapipe-selfie'.
import { createServer } from 'vite'
import { chromium } from 'playwright-core'
import { writeFileSync } from 'node:fs'
const noWebGPU = process.argv.includes('--nowebgpu')
const root = new URL('../', import.meta.url).pathname
const server = await createServer({ configFile: `${root}vite.config.js`, server: { port: 0 }, logLevel: 'error' })
await server.listen()
const origin = server.resolvedUrls.local[0].replace(/\/$/, '')
const browser = await chromium.launch({ channel: 'chrome', headless: true,
  args: ['--autoplay-policy=no-user-gesture-required', ...(noWebGPU ? ['--disable-features=WebGPU,WebGPUService', '--disable-webgpu'] : ['--enable-unsafe-webgpu'])] })
const page = await browser.newPage()
const logs = []
page.on('console', (m) => { const t = m.text(); if (/backend ready|error|Error|MediaPipe selfie/.test(t)) logs.push(t.slice(0, 200)) })
page.on('pageerror', (e) => logs.push('[pageerror] ' + e.message))
page.setDefaultTimeout(120000)
await page.goto(`${origin}/smoke.html`)
await page.waitForFunction(() => window.smoke)
const out = { gpu: await page.evaluate(() => window.smoke.gpu) }
out.pipeline = await page.evaluate(() => window.smoke.pipelineSmoke())
if (out.pipeline.thumb) { writeFileSync(`smoke-${noWebGPU ? 'webgl' : 'webgpu'}.jpg`, Buffer.from(out.pipeline.thumb.split(',')[1], 'base64')); delete out.pipeline.thumb }
if (!noWebGPU) out.masks = await page.evaluate(() => window.smoke.maskCompare())
out.logs = logs
console.log(JSON.stringify(out, null, 1))
await browser.close(); await server.close()
