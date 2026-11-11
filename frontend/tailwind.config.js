/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          dark: '#0f172a',
          card: '#1e293b',
          border: '#334155',
          up: '#10b981',    // Emerald-500 for color-blind safe Up
          down: '#ef4444',  // Red-500 for Down
          neutral: '#94a3b8',
          accent: '#3b82f6',
        }
      }
    },
  },
  plugins: [],
}

