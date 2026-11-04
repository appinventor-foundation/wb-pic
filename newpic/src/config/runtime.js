// Build-time settings (Create React App inlines REACT_APP_* variables at build).
// Defaults for `npm start` and `npm run build` are in .env.development / .env.production.

const withoutTrailingSlash = (url) => url.replace(/\/+$/, '');

// Training server (the node app). Empty means a browser-only app: no server
// training and ZIP datasets are read in the browser.
export const SERVER_URL = withoutTrailingSlash(process.env.REACT_APP_SERVER_URL || '');
export const SERVER_ENABLED = SERVER_URL !== '';

// Where the bundled transfer models are read from. Defaults to the copy shipped
// with the app (public/models, see scripts/copy-models.js).
export const MODELS_URL = withoutTrailingSlash(
  process.env.REACT_APP_MODELS_URL || `${process.env.PUBLIC_URL || ''}/models`
);
