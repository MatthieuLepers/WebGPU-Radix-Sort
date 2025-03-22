import { defineConfig } from 'vite';

export default defineConfig({
  base: process.env.NODE_ENV === 'production' ? '/dist/example/' : '/',
  build: {
    outDir: 'dist/example',
  },
});
