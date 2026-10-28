# Deploying Manobal-AI

Vercel and Netlify host the frontend. The API and MongoDB also need to run on publicly reachable services; a deployed website cannot connect to `localhost:8001` on your laptop.

## Frontend on Vercel or Netlify

1. Import this GitHub repository.
2. Set the frontend project/root directory to `frontend` (Netlify can use the included root `netlify.toml`).
3. Use build command `npm run build` and output directory `dist`.
4. Add the environment variable `VITE_API_BASE` with the deployed API root, including `/api`, for example `https://YOUR-BACKEND-HOST/api`.
5. Redeploy after adding or changing the variable. Vite embeds it at build time.

Vercel already has an SPA rewrite in `frontend/vercel.json`; the root `netlify.toml` provides Netlify's SPA fallback.

## Backend and database

Deploy the `backend` directory as a Docker service (its Dockerfile is `backend/Dockerfile`) and attach a persistent MongoDB service such as MongoDB Atlas. Configure these backend environment variables:

```text
MONGO_URL=<MongoDB connection string>
DB_NAME=manobal_ai
AUTH_SECRET_KEY=<a long, private random secret>
MODEL_ARTIFACT_DIR=/app/artifacts
CORS_ORIGINS=https://YOUR-FRONTEND-DOMAIN
ENABLE_DEMO_ACCOUNTS=true
```

For a custom domain, set `CORS_ORIGINS` to that exact origin too. Do not include a trailing slash. `backend/artifacts/` is included in the backend Docker build and contains the model files.

The current demo-login shortcut accepts any non-empty username and password when `ENABLE_DEMO_ACCOUNTS=true`. This is appropriate only for a presentation using synthetic demo records: anyone who can reach the deployed site can enter the selected demo role. For a private or real-data deployment, set it to `false` and use normal accounts instead.

## Quick deployment check

- Open `https://YOUR-BACKEND-HOST/api/` and confirm it returns `{"message":"Manobal-AI API","status":"ready"}`.
- Confirm the frontend's `VITE_API_BASE` points to that backend URL plus `/api`.
- Confirm the deployed backend's `CORS_ORIGINS` exactly matches the browser's frontend origin.
- Redeploy the frontend after environment variable changes, then try logging in and loading demo records.
