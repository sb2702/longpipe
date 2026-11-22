// `test` for the e2e specs. Locally / on CI it's Playwright's own. Projects with
// `metadata.browserstack` (tests/e2e/browserstack.config.mjs) get their
// `browser` from BrowserStack instead: a desktop browser or a real iOS device
// (`<engine>.connect`), or a real Android device's Chrome (`_android`). One
// BrowserStack session per worker; each test still gets a fresh page.
//
// Credentials come only from BROWSERSTACK_USERNAME / BROWSERSTACK_ACCESS_KEY.
import { test as base } from '@playwright/test'
import { createRequire } from 'node:module'

const PW_VERSION = createRequire(import.meta.url)('@playwright/test/package.json').version
const BUILD = process.env.BROWSERSTACK_BUILD_NAME ?? `longpipe e2e ${new Date().toISOString().slice(0, 16)}`

function wsEndpoint(caps, name) {
  const all = {
    ...caps,
    'browserstack.username': process.env.BROWSERSTACK_USERNAME,
    'browserstack.accessKey': process.env.BROWSERSTACK_ACCESS_KEY,
    'browserstack.local': 'true',
    ...(process.env.BROWSERSTACK_LOCAL_IDENTIFIER ? { 'browserstack.localIdentifier': process.env.BROWSERSTACK_LOCAL_IDENTIFIER } : {}),
    'browserstack.acceptInsecureCerts': 'true',   // the example is served over HTTPS with a throwaway cert
    'browserstack.debug': 'true',
    'browserstack.console': 'errors',
    'client.playwrightVersion': PW_VERSION,
    project: 'longpipe',
    build: BUILD,
    name,
  }
  return `wss://cdp.browserstack.com/playwright?caps=${encodeURIComponent(JSON.stringify(all))}`
}

// Mark the BrowserStack session passed/failed on its dashboard.
export const setSessionStatus = (page, status, reason) => page.evaluate(() => {},
  `browserstack_executor: ${JSON.stringify({ action: 'setSessionStatus', arguments: { status, reason: String(reason).slice(0, 250) } })}`).catch(() => {})

// Real devices allow ONE browser context per session, and iOS one tab (and
// _android gives a BrowserContext, not a Browser): present the session as a
// browser whose newContext() hands out that one context, whose newPage() hands
// out the one page. Tests navigate it afresh; "closing" either is a no-op.
function singleContextBrowser(context, close) {
  const newPage = context.newPage.bind(context)
  let page = null
  context.newPage = async () => {
    if (!page || page.isClosed()) page = await newPage()
    return page
  }
  context.close = async () => {}
  return {
    _connection: {},   // read by Playwright's context fixture
    newContext: async () => context,
    contexts: () => [context],
    version: () => 'device',
    isConnected: () => true,
    close,
  }
}

const failures = []

export const test = base.extend({
  browser: [async ({ playwright, browserName, _browserOptions }, use, workerInfo) => {
    const bs = workerInfo.project.metadata?.browserstack
    if (!bs) {
      // Playwright's default (launch options come from the project via _browserOptions).
      const browser = await playwright[browserName].launch()
      await use(browser)
      await browser.close()
      return
    }
    if (!process.env.BROWSERSTACK_USERNAME || !process.env.BROWSERSTACK_ACCESS_KEY) {
      throw new Error('BrowserStack projects need BROWSERSTACK_USERNAME and BROWSERSTACK_ACCESS_KEY')
    }
    const ws = wsEndpoint(bs.caps, `${workerInfo.project.name} #${workerInfo.workerIndex}`)
    let browser
    if (bs.android) {
      const device = await playwright._android.connect(ws)
      await device.shell('am force-stop com.android.chrome')
      browser = singleContextBrowser(await device.launchBrowser({ ignoreHTTPSErrors: true }), () => device.close().catch(() => {}))
    } else if (bs.caps.realMobile) {
      const remote = await playwright[browserName].connect(ws)
      browser = singleContextBrowser(await remote.newContext({ ignoreHTTPSErrors: true }), () => remote.close().catch(() => {}))
    } else {
      browser = await playwright[browserName].connect(ws)
    }
    await use(browser)
    // Session status on the dashboard: failed if any test in this worker failed.
    const ctx = browser.contexts()[0] ?? await browser.newContext().catch(() => null)
    const page = ctx && (ctx.pages()[0] ?? await ctx.newPage().catch(() => null))
    if (page) await setSessionStatus(page, failures.length ? 'failed' : 'passed', failures.join('; ') || 'all passed')
    await browser.close().catch(() => {})
  }, { scope: 'worker', timeout: 240_000 }],

  page: async ({ page }, use, testInfo) => {
    const bs = testInfo.project.metadata?.browserstack
    if (bs?.caps.realMobile) {
      // The shared device context has no baseURL: resolve relative goto()s here.
      const goto = page.goto.bind(page)
      page.goto = (url, opts) => goto(new URL(url, testInfo.project.use.baseURL).href, opts)
    }
    await use(page)
    if (bs && testInfo.status !== testInfo.expectedStatus && testInfo.status !== 'skipped') {
      failures.push(`${testInfo.title}: ${testInfo.error?.message?.split('\n')[0] ?? testInfo.status}`)
    }
  },
})

export { expect } from '@playwright/test'
