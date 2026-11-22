// Starts BrowserStack Local (the tunnel the devices use to reach the local
// example server) for tests/e2e/browserstack.config.mjs; stops it afterwards.
// Skipped when BROWSERSTACK_LOCAL_RUNNING=1 (a tunnel you started yourself).
import browserstack from 'browserstack-local'

export default async function globalSetup() {
  if (process.env.BROWSERSTACK_LOCAL_RUNNING === '1') return
  const key = process.env.BROWSERSTACK_ACCESS_KEY
  if (!key) throw new Error('BROWSERSTACK_ACCESS_KEY is not set')
  const local = new browserstack.Local()
  const args = { key, force: 'true', forceLocal: 'true', logFile: 'tests/e2e/results-bs/browserstack-local.log' }
  if (process.env.BROWSERSTACK_LOCAL_IDENTIFIER) args.localIdentifier = process.env.BROWSERSTACK_LOCAL_IDENTIFIER
  await new Promise((resolve, reject) => local.start(args, (err) => (err ? reject(new Error(`BrowserStack Local: ${err.message}`)) : resolve())))
  console.log('BrowserStack Local tunnel up')
  return () => new Promise((resolve) => local.stop(() => resolve()))
}
