/**
 * Context-menu provider for the 'canvas' type (right-click on empty canvas space).
 *
 * Adapter interface (CtxCanvasAdapter):
 *   getImageUuid()  → string | null
 */

import { registerContextProvider } from '../registry.js';
import { asyncPanel } from '../menu.js';
import { openSimilarImagesModal, getImageHashMax, getImageDefaultThreshold } from '/app-static/js/ui/similarSearch.js';
import { buildSimilarPanel } from '../similarPanelBuilder.js';


registerContextProvider('canvas', (adapters, ctx) => {
  const imageUuid = adapters[0]?.getImageUuid?.() ?? null;
  if (!imageUuid) return [];

  const items = [
    {
      label: 'Search for similar images…',
      submenu: () => asyncPanel('200px', async () => {
        const hashMax = await getImageHashMax();
        const defaultThreshold = await getImageDefaultThreshold(hashMax);
        return buildSimilarPanel((threshold) => {
          ctx.close();
          openSimilarImagesModal({ imageUuid, threshold });
        }, defaultThreshold, hashMax);
      }),
    },
  ];

  return [{ items }];
});
