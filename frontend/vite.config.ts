import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        host: true, // Bind to 0.0.0.0 so cloudflared can reach it
        allowedHosts: [
            'footwear-fin-decades-vitamins.trycloudflare.com',
        ],
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
            },
        },
    },
})
