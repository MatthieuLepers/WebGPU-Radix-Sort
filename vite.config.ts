import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    target: 'ESNext',
    sourcemap: true,
    lib: {
      entry: 'src/index.ts',
      name: 'RadixSort',
      fileName: (format) => `radix-sort.${format}.js`,
      formats: ['es', 'cjs', 'umd'],
    },
  },
});
