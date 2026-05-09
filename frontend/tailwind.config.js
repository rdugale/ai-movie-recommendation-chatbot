/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cinema: {
          dark:   "#0d0d0d",
          card:   "#1a1a2e",
          accent: "#e94560",
          gold:   "#f5a623",
          dim:    "#6b7280",
          border: "#2a2a3e",
        },
      },
      fontFamily: {
        display: ["'Playfair Display'", "serif"],
        body:    ["'Inter'", "sans-serif"],
      },
    },
  },
  plugins: [],
};

