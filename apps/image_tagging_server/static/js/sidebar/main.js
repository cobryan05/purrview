import { getCurrentImageName } from '/app-static/js/canvas/state.js';

document.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.tab-button');
  const tabContents = document.querySelectorAll('.tab-content');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const tabName = tab.getAttribute('data-tab');
      tabContents.forEach(tc => {
        tc.style.display = tc.id === tabName ? 'block' : 'none';
      });
    });
  });

  refreshModelList();
  document.getElementById('recognize-button').addEventListener('click', recognizeImage);
});

async function refreshModelList() {
  try {
    const res = await fetch('/api/list-models');
    const data = await res.json();
    const modelListContainer = document.getElementById('model-list');

    if (!modelListContainer) {
      console.error("modelListContainer is null");
      return;
    }

    modelListContainer.innerHTML = '';
    data.models.forEach(model => {
      const label = document.createElement('label');
      label.style.display = 'block';

      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'model';
      radio.value = model;

      label.appendChild(radio);
      label.appendChild(document.createTextNode(` ${model}`));
      modelListContainer.appendChild(label);
    });

    const firstRadio = modelListContainer.querySelector('input[type="radio"]');
    if (firstRadio) firstRadio.checked = true;
  } catch (err) {
    console.error('Failed to fetch models:', err);
  }
}

async function recognizeImage() {
  try {
    const selected = document.querySelector('input[name="model"]:checked');
    if (!selected) {
      alert('Please select a model');
      return;
    }

    const modelName = selected.value;
    const imageName = getCurrentImageName();
    if (!imageName) {
      alert('No image loaded');
      return;
    }

    const imageUrl = `/images/${imageName}.jpg`;
    const res = await fetch(imageUrl);
    if (!res.ok) throw new Error(`Failed to load image from ${imageUrl}`);

    const blob = await res.blob();

    // Create FormData and append fields as strings
    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('conf_thresh', '0.25');           // string is fine
    formData.append('return_annotated', 'false');     // string is fine
    formData.append('image', blob, `${imageName}.png`); // optional: send PNG filename extension if image is PNG

    const response = await fetch('/api/recognize', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Recognition API error: ${response.status} ${text}`);
    }

    const result = await response.json();
    console.log('Recognize response:', result);
  } catch (err) {
    console.error('Failed to recognize image:', err);
  }
}
