// filesApi.js

/* ---- File Listing --- */
export async function fetchImageList(path) {
  const res = await fetch('/api/images/list', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: path }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch image list');
  }
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch image list');
  }

  return data.files;
}

/* ---- File Downloading --- */
export async function fetchImage(path) {
  const imageRes = await fetch(`/api/images/get`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: path }),
  });
  if (!imageRes.ok) throw new Error(`Failed to load image via API for ${path}`);

  const imageJson = await imageRes.json();
  if (imageJson.status !== 'success' || !imageJson.content) {
    throw new Error(`Invalid image API response for ${path}`);
  }

  // === Decode image ===
  const img = new Image();
  img.src = `data:${imageJson.mime_type};base64,${imageJson.content}`;
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () =>
      reject(new Error(`Failed to decode base64 image for ${path}`));
  });
  return { image: img, bboxes: imageJson.boxes || [] };
}
