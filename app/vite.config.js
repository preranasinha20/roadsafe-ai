import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' lets the same build run from a sub-path (GitHub Pages) or a domain root (Vercel/Netlify)
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { chunkSizeWarningLimit: 1200 },
})
