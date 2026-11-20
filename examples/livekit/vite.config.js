// Vite config for the LiveKit example: the vanilla-WebRTC example's config
// (Longpipe compiled from this repo's src/, or dist/ with LONGPIPE=dist), plus
//   - @longpipe/livekit from packages/livekit/src,
//   - the e2e clip/weights from ../vanilla-webrtc/public (npm run e2e:assets),
//   - GET /token?identity=…: a LiveKit access token, minted with the dev key.
//
//   LIVEKIT_URL      ws URL the page connects to   (default ws://localhost:7880)
//   LIVEKIT_API_KEY / LIVEKIT_API_SECRET           (default devkey / secret: `livekit-server --dev`)
import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { AccessToken } from 'livekit-server-sdk'
import vanilla from '../vanilla-webrtc/vite.config.js'

const HERE = fileURLToPath(new URL('.', import.meta.url))
const REPO = resolve(HERE, '../..')
const useDist = process.env.LONGPIPE === 'dist'

const tokenEndpoint = {
  name: 'livekit-dev-token',
  configureServer(server) {
    server.middlewares.use('/token', async (req, res) => {
      const q = new URL(req.url, 'http://x').searchParams
      const at = new AccessToken(process.env.LIVEKIT_API_KEY ?? 'devkey', process.env.LIVEKIT_API_SECRET ?? 'secret', {
        identity: q.get('identity') ?? 'guest', ttl: '1h',
      })
      at.addGrant({ roomJoin: true, room: q.get('room') ?? 'longpipe', canPublish: true, canSubscribe: true })
      res.setHeader('content-type', 'application/json')
      res.end(JSON.stringify({ url: process.env.LIVEKIT_URL ?? 'ws://localhost:7880', token: await at.toJwt() }))
    })
  },
}

export default defineConfig((env) => {
  const base = vanilla(env)
  return {
    ...base,
    root: HERE,
    publicDir: resolve(REPO, 'examples/vanilla-webrtc/public'),
    plugins: [...base.plugins, tokenEndpoint],
    resolve: {
      alias: [
        { find: /^@longpipe\/livekit$/, replacement: resolve(REPO, 'packages/livekit/src/index.ts') },
        { find: /^longpipe\/audio$/, replacement: useDist ? resolve(REPO, 'dist/audio.js') : resolve(REPO, 'src/audio.ts') },
        ...base.resolve.alias,
      ],
    },
    server: { ...base.server, port: Number(process.env.PORT ?? 5290) },
    optimizeDeps: { exclude: ['longpipe', '@longpipe/livekit'] },
  }
})
