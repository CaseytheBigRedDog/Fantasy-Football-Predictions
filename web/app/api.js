// Small helper used by every component to talk to the FastAPI service.
export async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("Could not reach the API. Is the FastAPI service running (uvicorn)?");
  }
  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* the body was not JSON */
    }
    if (response.status >= 500 && !detail) {
      detail = "The API did not answer. Is the FastAPI service running (uvicorn)?";
    }
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json();
}

export const fmt = (n) => (n === null || n === undefined ? "–" : Number(n).toFixed(1));
