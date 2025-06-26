import { getCurrentImageName } from '/app-static/js/canvas/state.js';
import { addRecognizedBoxes } from '/app-static/js/canvas/io.js';

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-header').forEach(tabHeader => {
    const tabs = tabHeader.querySelectorAll('.tab-button');
    const tabContainer = tabHeader.parentElement;
    const tabPanes = tabContainer.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const tabName = tab.getAttribute('data-tab');
        tabPanes.forEach(tc => {
          tc.classList.toggle('active', tc.id === tabName);
        });
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
    if (firstRadio) {
      firstRadio.checked = true;
    }
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
    if (result.status === 'success') {
      addRecognizedBoxes(result.results);
    }
  } catch (err) {
    console.error('Failed to recognize image:', err);
  }
}

export function openInspectorTab() {
  // Scope tab switching to the sidebar tab group only
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;

  const tabs = sidebar.querySelectorAll('.tab-button');
  const panes = sidebar.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    const tabName = tab.getAttribute('data-tab');
    const isInspector = tabName === 'tab-inspector';
    tab.classList.toggle('active', isInspector);
  });

  panes.forEach(pane => {
    pane.classList.toggle('active', pane.id === 'tab-inspector');
  });

  document.getElementById('labelInput')?.focus();

  const inspector = document.getElementById('tab-inspector');
  if (inspector) {
    inspector.style.outline = '2px solid #ff0033';
    setTimeout(() => inspector.style.outline = '', 1000);
  }
}
