/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'Roboto', 'sans-serif'],
        poppins: ['Poppins', 'sans-serif']
      }
    }
  },
  // Important: comme vous utilisez déjà Material-UI, ajoutez ceci pour éviter les conflits
  important: '#root',
  // Désactiver le preflight si vous utilisez Material-UI
  corePlugins: {
    preflight: false
  },
  plugins: []
};
